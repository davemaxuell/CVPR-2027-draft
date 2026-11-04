"""Observable-only zone records, saved-view layouts and blame-bounded retention.

This module imports neither a map nor simulator labels. Callers provide predicted keep,
blame/progress scores, observed headings and odometry. It never chooses a driving action.
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field


def distance(a, b):
    return math.sqrt(sum((float(x)-float(y))**2 for x,y in zip(a,b)))


def angle_distance(a,b):
    return abs((a-b+math.pi)%(2*math.pi)-math.pi)


def diverse_views(decisions, maximum=3):
    """Commitment view first; every additional heading differs by at least 30 degrees."""
    if not decisions:
        return []
    selected = [decisions[-1]]
    remaining = decisions[:-1]
    while remaining and len(selected)<maximum:
        best = max(remaining,key=lambda x:(min(angle_distance(x['heading'],y['heading']) for y in selected),-x['t']))
        if min(angle_distance(best['heading'],x['heading']) for x in selected)<math.pi/6-1e-8:
            break
        selected.append(best)
        remaining = [x for x in remaining if x['t']!=best['t']]
    return [int(x.get('frame',x.get('sim_step',x['t']))) for x in selected]


@dataclass
class Zone:
    identifier: int
    first: dict
    decisions: list = field(default_factory=list)
    closed_at: int | None = None
    exit_heading: float | None = None

    @property
    def commitment(self):
        return self.decisions[-1]

    def record(self):
        commit = self.commitment
        return dict(id=self.identifier,first_decision=self.first['t'],commitment_decision=commit['t'],
                    commitment_frame=int(commit.get('frame',commit.get('sim_step',commit['t']))),
                    closed_at=self.closed_at,views=diverse_views(self.decisions),
                    opening_headings=list(commit.get('opening_headings',self.first.get('opening_headings',[]))),
                    exit_heading=self.exit_heading,commitment_heading=commit['heading'],position=list(commit['position']),
                    path_length=float(commit.get('path_length',0)),
                    keep_score=max(float(x.get('keep_score',0)) for x in self.decisions),
                    part=int(commit.get('part',0)))


class ZoneBuilder:
    """1.5-m membership / >2-m closure hysteresis, one record per closed zone."""
    def __init__(self):
        self.active = None
        self.next_identifier = 0

    def update(self,decision,candidate):
        closed = None
        if self.active is not None:
            displacement = distance(decision['position'],self.active.first['position'])
            if displacement>2.0:
                self.active.closed_at = int(decision['t'])
                delta = [float(a)-float(b) for a,b in zip(decision['position'],self.active.commitment['position'])]
                if math.hypot(delta[0],delta[2])>1e-8:
                    self.active.exit_heading = math.atan2(-delta[0],-delta[2])
                closed = self.active.record()
                self.active = None
            elif displacement<=1.5:
                self.active.decisions.append(dict(decision))
        if self.active is None and candidate:
            self.active = Zone(self.next_identifier,dict(decision),[dict(decision)])
            self.next_identifier += 1
        return closed

    def finish(self,decision_count):
        if self.active is None:
            return None
        self.active.closed_at = int(decision_count)
        record = self.active.record()
        self.active = None
        return record


def inject_saved_places(pieces,records,max_records=3,max_views=2):
    """Actual past frame indices only; no labels/parts/poses enter the drive prompt."""
    insertion = [('text','\nSaved places: ')]
    if not records:
        insertion.append(('text','<empty>.\n'))
    for i,record in enumerate(records[:max_records]):
        insertion.append(('text',f'[place {i+1}] '))
        insertion.extend(('image',int(frame),'memory') for frame in record['views'][:max_views])
        insertion.append(('text','; '))
    result = list(pieces)
    # The first user turn starts with the navigation instruction, after the system turn.
    first_user = next(i for i,p in enumerate(result) if p==('open','user'))
    if result[first_user+1][0]!='text':
        raise ValueError('Unexpected drive layout: no instruction text after the first user role.')
    return result[:first_user+2]+insertion+result[first_user+2:]


@dataclass
class Retained:
    record: dict
    blame: float = 0.0
    low_history: deque = field(default_factory=deque)
    cleared: bool = False


class Retention:
    def __init__(self,k=3,epsilon=0.05,horizon=3,minimum_age=3,dormant_cap=8,progress_confidence=0.8,gamma=0.97):
        self.k,self.epsilon,self.horizon,self.minimum_age = k,epsilon,horizon,minimum_age
        self.dormant_cap,self.progress_confidence,self.gamma = dormant_cap,progress_confidence,gamma
        self.records = {}
        self.events = []
        self.last_t = -1

    def add(self,record):
        if record['id'] in self.records:
            raise ValueError('A zone must be represented by exactly one record.')
        self.records[record['id']] = Retained(dict(record),low_history=deque(maxlen=self.horizon))

    def suspect(self,item,t):
        return item.blame>=self.epsilon or t-item.record['commitment_decision']<self.minimum_age

    def erv(self,item,path_length):
        back = max(0.0,path_length-item.record['path_length'])
        return item.blame*item.record['keep_score']*self.gamma**back

    def update(self,t,blames,progress_probabilities,path_length):
        """Keep unconfirmed records and suspects; cap only evidence-cleared dormant records."""
        if t<=self.last_t:
            raise ValueError('Retention updates must advance one decision at a time.')
        self.last_t = t
        for identifier,item in self.records.items():
            if identifier not in blames:
                # An unqueried record has no exonerating evidence and cannot be deleted.
                item.low_history.append((t,False))
                item.cleared = False
                continue
            item.blame = float(blames[identifier])
            if not math.isfinite(item.blame) or not 0<=item.blame<=1:
                raise ValueError('Invalid model blame probability.')
            item.low_history.append((t,item.blame<self.epsilon))
            beyond = sum(float(p) for i,p in enumerate(progress_probabilities) if i>item.record['part'])
            item.cleared = (not self.suspect(item,t) and len(item.low_history)==self.horizon and
                            all(low for _,low in item.low_history) and
                            [step for step,_ in item.low_history]==list(range(t-self.horizon+1,t+1)) and
                            beyond>=self.progress_confidence)
        dormant = [x for x in self.records.values() if x.cleared and not self.suspect(x,t)]
        dormant.sort(key=lambda x:(self.erv(x,path_length),x.record['commitment_decision']))
        for item in dormant[:max(0,len(dormant)-self.dormant_cap)]:
            identifier = item.record['id']
            self.events.append(dict(t=t,record_id=identifier,event='delete_evidence_cleared_dormant',
                                    blame=item.blame,path_length=path_length))
            del self.records[identifier]
        ordered = sorted(self.records.values(),key=lambda x:(-self.erv(x,path_length),-x.record['commitment_decision']))
        return [x.record for x in ordered[:self.k]]

    def summary(self,t):
        return dict(stored=len(self.records),suspects=sum(self.suspect(x,t) for x in self.records.values()),
                    dormant=sum(x.cleared and not self.suspect(x,t) for x in self.records.values()),
                    pending_clearance=sum(not x.cleared and not self.suspect(x,t) for x in self.records.values()))


class ObservableMemory:
    """Prefreeze and baseline saved-place rules, using observations/odometry alone."""
    def __init__(self,mode='empty',k=3,every_n=3.0,seed=0):
        if mode not in {'empty','every_n','random'}:
            raise ValueError('This observable baseline does not implement a learned rule.')
        self.mode,self.k,self.every_n = mode,k,every_n
        self.rng = random.Random(seed)
        self.builder = ZoneBuilder()
        self.selected,self.all_records = [],[]
        self.last_save_distance = 0.0
        self.seen = 0

    def observe_decision(self,decision):
        closed = self.builder.update(decision,True)
        if closed is not None:
            self._save(closed)

    def _save(self,record):
        self.all_records.append(record)
        if self.mode=='empty':
            return
        if self.mode=='every_n':
            if record['path_length']-self.last_save_distance+1e-8<self.every_n:
                return
            self.last_save_distance = record['path_length']
            self.selected = (self.selected+[record])[-self.k:]
        elif self.mode=='random':
            self.seen += 1
            if len(self.selected)<self.k:
                self.selected.append(record)
            else:
                index = self.rng.randrange(self.seen)
                if index<self.k:
                    self.selected[index] = record

    def inputs(self,before_frame):
        output = []
        for record in sorted(self.selected,key=lambda x:x['commitment_frame']):
            views = [int(x) for x in record['views'][:2]]
            if any(not 0<=x<before_frame for x in views):
                raise RuntimeError('A model memory input is not an already-past frame.')
            output.append(dict(views=views))
        return output
