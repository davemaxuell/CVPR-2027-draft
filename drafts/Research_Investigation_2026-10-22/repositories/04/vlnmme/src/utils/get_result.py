#!/usr/bin/env python3
import os
import re
from pathlib import Path
from collections import defaultdict

def parse_valid_file(file_path):
    """Parse a valid.txt file and extract R2R, REVERIE, and OBJECTNAV metrics."""
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
        
        results = {}
        
        # Extract R2R data - all metrics
        r2r_pattern = r'Env name: R2R\.val_unseen_subset, action_steps: ([\d.]+), steps: ([\d.]+), lengths: ([\d.]+), nav_error: ([\d.]+), oracle_error: ([\d.]+), sr: ([\d.]+), oracle_sr: ([\d.]+), spl: ([\d.]+), nDTW: ([\d.]+), SDTW: ([\d.]+), CLS: ([\d.]+)'
        r2r_match = re.search(r2r_pattern, content)
        if r2r_match:
            results['R2R'] = {
                'action_steps': float(r2r_match.group(1)),
                'nav_error': float(r2r_match.group(4)),
                'oracle_error': float(r2r_match.group(5)),
                'sr': float(r2r_match.group(6)),
                'oracle_sr': float(r2r_match.group(7)),
                'spl': float(r2r_match.group(8)),
                'nDTW': float(r2r_match.group(9)),
                'SDTW': float(r2r_match.group(10)),
                'CLS': float(r2r_match.group(11))
            }
        
        # Extract REVERIE data - all metrics
        reverie_pattern = r'Env name: REVERIE\.val_unseen_subset, action_steps: ([\d.]+), steps: ([\d.]+), lengths: ([\d.]+), nav_error: ([\d.]+), oracle_error: ([\d.]+), sr: ([\d.]+), oracle_sr: ([\d.]+), spl: ([\d.]+), nDTW: ([\d.]+), SDTW: ([\d.]+), CLS: ([\d.]+)'
        reverie_match = re.search(reverie_pattern, content)
        if reverie_match:
            results['REVERIE'] = {
                'action_steps': float(reverie_match.group(1)),
                'nav_error': float(reverie_match.group(4)),
                'oracle_error': float(reverie_match.group(5)),
                'sr': float(reverie_match.group(6)),
                'oracle_sr': float(reverie_match.group(7)),
                'spl': float(reverie_match.group(8)),
                'nDTW': float(reverie_match.group(9)),
                'SDTW': float(reverie_match.group(10)),
                'CLS': float(reverie_match.group(11))
            }
        
        # Extract OBJECTNAV data - all metrics
        objectnav_pattern = r'Env name: OBJECTNAV\.val_unseen_subset, action_steps: ([\d.]+), steps: ([\d.]+), lengths: ([\d.]+), nav_error: ([\d.]+), oracle_error: ([\d.]+), sr: ([\d.]+), oracle_sr: ([\d.]+), spl: ([\d.]+), nDTW: ([\d.]+), SDTW: ([\d.]+), CLS: ([\d.]+)'
        objectnav_match = re.search(objectnav_pattern, content)
        if objectnav_match:
            results['OBJECTNAV'] = {
                'action_steps': float(objectnav_match.group(1)),
                'nav_error': float(objectnav_match.group(4)),
                'oracle_error': float(objectnav_match.group(5)),
                'sr': float(objectnav_match.group(6)),
                'oracle_sr': float(objectnav_match.group(7)),
                'spl': float(objectnav_match.group(8)),
                'nDTW': float(objectnav_match.group(9)),
                'SDTW': float(objectnav_match.group(10)),
                'CLS': float(objectnav_match.group(11))
            }
        
        return results
    
    except (FileNotFoundError, IOError):
        return {}

def scan_outputs_folder(outputs_path="./outputs/"):
    """Scan the outputs folder and extract all evaluation data."""
    data = defaultdict(dict)
    
    outputs_dir = Path(outputs_path)
    if not outputs_dir.exists():
        print(f"Error: {outputs_path} does not exist!")
        return data
    
    # Iterate through agent folders
    for agent_folder in outputs_dir.iterdir():
        if not agent_folder.is_dir():
            continue
        
        agent_name = agent_folder.name
        
        # Iterate through model folders within each agent folder
        for model_folder in agent_folder.iterdir():
            if not model_folder.is_dir():
                continue
            
            model_name = model_folder.name
            valid_file = model_folder / "valid.txt"
            
            # Parse the valid.txt file if it exists
            results = parse_valid_file(valid_file)
            data[agent_name][model_name] = results
    
    return data

def format_metric(value, decimal_places=2):
    """Format metric value, return empty string if None."""
    if value is None:
        return ""
    return f"{value:.{decimal_places}f}"

def generate_html_table(data):
    """Generate HTML table from the extracted data."""
    if not data:
        return "<p>No data found!</p>"
    
    # Define metrics in the order they should appear
    metrics = ['action_steps', 'nav_error', 'oracle_error', 'sr', 'oracle_sr', 'spl', 'nDTW', 'SDTW', 'CLS']
    metric_names = ['Steps', 'Nav Error', 'Oracle Error', 'SR', 'Oracle SR', 'SPL', 'nDTW', 'SDTW', 'CLS']
    
    # Collect all unique agents, models, and tasks
    all_agents = sorted(data.keys())
    all_models = set()
    all_tasks = set()
    
    for agent_data in data.values():
        all_models.update(agent_data.keys())
        for model_results in agent_data.values():
            all_tasks.update(model_results.keys())
    
    all_models = sorted(all_models)
    all_tasks = sorted(all_tasks, key=lambda x: ['R2R', 'REVERIE', 'OBJECTNAV'].index(x) if x in ['R2R', 'REVERIE', 'OBJECTNAV'] else 999)
    
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Navigation Agent Evaluation Results</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 30px;
                max-width: 100%;
                overflow-x: auto;
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
                font-size: 28px;
            }
            .summary {
                background: #e3f2fd;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 25px;
                text-align: center;
                color: #1565c0;
            }
            .filters {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 25px;
                border: 1px solid #dee2e6;
            }
            .filters h3 {
                margin-top: 0;
                color: #495057;
                font-size: 18px;
            }
            .filter-group {
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                align-items: center;
                margin-bottom: 15px;
            }
            .filter-group label {
                font-weight: 600;
                color: #495057;
                min-width: 80px;
            }
            .filter-checkboxes {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }
            .checkbox-item {
                display: flex;
                align-items: center;
                background: white;
                padding: 8px 12px;
                border: 1px solid #ced4da;
                border-radius: 5px;
                transition: all 0.2s;
                cursor: pointer;
            }
            .checkbox-item:hover {
                background: #e9ecef;
                border-color: #6c757d;
            }
            .checkbox-item input[type="checkbox"] {
                margin-right: 8px;
                cursor: pointer;
            }
            .checkbox-item.checked {
                background: #d4edda;
                border-color: #28a745;
                color: #155724;
                font-weight: 500;
            }
            .filter-actions {
                display: flex;
                gap: 10px;
                margin-top: 15px;
            }
            .btn {
                background: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                transition: background-color 0.2s;
            }
            .btn:hover {
                background: #0056b3;
            }
            .btn-secondary {
                background: #6c757d;
            }
            .btn-secondary:hover {
                background: #545b62;
            }
            .sort-info {
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 15px;
                color: #856404;
                font-size: 14px;
            }
            .results-info {
                background: #d1ecf1;
                border: 1px solid #bee5eb;
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 15px;
                color: #0c5460;
                font-size: 14px;
                text-align: center;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .task-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-align: center;
                font-weight: bold;
                font-size: 14px;
                padding: 12px 8px;
            }
            .r2r-header {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            }
            .reverie-header {
                background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            }
            .objectnav-header {
                background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            }
            .metric-header {
                background: #f8f9fa;
                color: #495057;
                text-align: center;
                font-weight: 600;
                font-size: 12px;
                padding: 10px 6px;
                border-bottom: 2px solid #dee2e6;
                cursor: pointer;
                user-select: none;
                transition: background-color 0.2s;
            }
            .metric-header:hover {
                background: #e9ecef;
            }
            .metric-header.sortable::after {
                content: " ↕️";
                font-size: 10px;
                opacity: 0.6;
            }
            .metric-header.sort-asc::after {
                content: " ↑";
                color: #28a745;
                font-weight: bold;
            }
            .metric-header.sort-desc::after {
                content: " ↓";
                color: #dc3545;
                font-weight: bold;
            }
            .agent-model-header {
                background: #343a40;
                color: white;
                text-align: center;
                font-weight: bold;
                padding: 12px;
                font-size: 14px;
                cursor: pointer;
                user-select: none;
                transition: background-color 0.2s;
            }
            .agent-model-header:hover {
                background: #495057;
            }
            .agent-model-header::after {
                content: " ↕️";
                font-size: 10px;
                opacity: 0.6;
            }
            th, td {
                border: 1px solid #dee2e6;
                padding: 10px 8px;
                text-align: center;
            }
            td {
                font-size: 13px;
            }
            .agent-cell {
                background: #f8f9fa;
                font-weight: 600;
                color: #495057;
            }
            .model-cell {
                background: #e9ecef;
                font-weight: 500;
                color: #6c757d;
            }
            .metric-cell {
                background: white;
            }
            .empty-cell {
                background: #ffeaa7;
                color: #636e72;
                font-style: italic;
            }
            .best-metric {
                background: #d4edda !important;
                font-weight: bold;
                color: #155724;
            }
            .worst-metric {
                background: #f8d7da !important;
                font-weight: bold;
                color: #721c24;
            }
            tr:hover td {
                background-color: #f1f3f4;
            }
            tr.best-row td.metric-cell {
                box-shadow: inset 3px 0 0 #28a745;
            }
            .hidden-row {
                display: none;
            }
            .footer {
                margin-top: 20px;
                text-align: center;
                color: #6c757d;
                font-size: 12px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Navigation Agent Evaluation Results</h1>
    """
    
    # Add summary
    total_agents = len(all_agents)
    total_models = len(all_models)
    total_tasks = len(all_tasks)
    html += f"""
            <div class="summary">
                📊 Found <strong>{total_agents}</strong> agents, <strong>{total_models}</strong> models, <strong>{total_tasks}</strong> tasks<br>
                🔍 Use the filters below to focus on specific combinations
            </div>
            
            <div class="filters">
                <h3>🔍 Filter Results</h3>
                
                <div class="filter-group">
                    <label>Tasks:</label>
                    <div class="filter-checkboxes" id="taskFilters">
    """
    
    # Add task filter checkboxes
    for task in all_tasks:
        html += f"""
                        <div class="checkbox-item" onclick="toggleCheckbox('task_{task}')">
                            <input type="checkbox" id="task_{task}" checked onchange="applyFilters()">
                            <span>{task}</span>
                        </div>
        """
    
    html += """
                    </div>
                </div>
                
                <div class="filter-group">
                    <label>Agents:</label>
                    <div class="filter-checkboxes" id="agentFilters">
    """
    
    # Add agent filter checkboxes
    for agent in all_agents:
        html += f"""
                        <div class="checkbox-item" onclick="toggleCheckbox('agent_{agent}')">
                            <input type="checkbox" id="agent_{agent}" checked onchange="applyFilters()">
                            <span>{agent}</span>
                        </div>
        """
    
    html += """
                    </div>
                </div>
                
                <div class="filter-group">
                    <label>Models:</label>
                    <div class="filter-checkboxes" id="modelFilters">
    """
    
    # Add model filter checkboxes
    for model in all_models:
        html += f"""
                        <div class="checkbox-item" onclick="toggleCheckbox('model_{model}')">
                            <input type="checkbox" id="model_{model}" checked onchange="applyFilters()">
                            <span>{model}</span>
                        </div>
        """
    
    html += """
                    </div>
                </div>
                
                <div class="filter-actions">
                    <button class="btn" onclick="selectAllFilters()">Select All</button>
                    <button class="btn btn-secondary" onclick="clearAllFilters()">Clear All</button>
                    <button class="btn btn-secondary" onclick="resetFilters()">Reset</button>
                </div>
            </div>
            
            <div class="results-info" id="resultsInfo">
                Showing all results
            </div>
            
            <div class="sort-info" id="sortInfo">
                🔄 Click on any column header to sort by that metric. Higher values are better for SR, Oracle SR, SPL, nDTW, SDTW, CLS. Lower values are better for Steps, Nav Error, Oracle Error.
            </div>
            
            <table id="resultsTable">
                <thead>
                    <tr>
                        <th class="agent-model-header" rowspan="2" onclick="sortTable(0, 'string')">Agent</th>
                        <th class="agent-model-header" rowspan="2" onclick="sortTable(1, 'string')">Model</th>
    """
    
    # Add task headers dynamically
    for task in all_tasks:
        task_class = f"{task.lower()}-header"
        html += f'<th class="task-header {task_class}" colspan="9" data-task="{task}">{task}</th>'
    
    html += "</tr><tr>"
    
    # Add metric headers for each available task
    col_index = 2
    for task in all_tasks:
        for i, metric_name in enumerate(metric_names):
            metric_key = metrics[i]
            # Determine if higher or lower is better
            higher_better = metric_key in ['sr', 'oracle_sr', 'spl', 'nDTW', 'SDTW', 'CLS']
            better_type = 'higher' if higher_better else 'lower'
            html += f'<th class="metric-header sortable" data-task="{task}" onclick="sortTable({col_index}, \'number\', \'{better_type}\')">{metric_name}</th>'
            col_index += 1
    
    html += "</tr></thead><tbody>"
    
    # Sort agents and models for consistent output
    for agent_name in sorted(data.keys()):
        agent_data = data[agent_name]
        
        for model_name in sorted(agent_data.keys()):
            model_results = agent_data[model_name]
            
            html += f'<tr data-agent="{agent_name}" data-model="{model_name}">'
            html += f'<td class="agent-cell">{agent_name}</td>'
            html += f'<td class="model-cell">{model_name}</td>'
            
            # Add metrics for each available task
            for task in all_tasks:
                task_data = model_results.get(task, {})
                for metric in metrics:
                    value = task_data.get(metric)
                    if value is not None:
                        formatted_value = format_metric(value)
                        html += f'<td class="metric-cell" data-task="{task}">{formatted_value}</td>'
                    else:
                        html += f'<td class="empty-cell" data-task="{task}">-</td>'
            
            html += "</tr>"
    
    html += """
                </tbody>
            </table>
            
            <div class="footer">
                Generated on """ + str(Path.cwd()) + """<br>
                💡 Empty cells (-) indicate missing data or incomplete evaluations<br>
                🎯 Click column headers to sort • Green highlights show best values • Red highlights show worst values<br>
                🔍 Use filters above to focus on specific tasks, agents, or models
            </div>
        </div>
        
        <script>
        let currentSort = {column: -1, direction: 'none'};
        
        function toggleCheckbox(checkboxId) {
            const checkbox = document.getElementById(checkboxId);
            checkbox.checked = !checkbox.checked;
            updateCheckboxStyle(checkbox);
            applyFilters();
        }
        
        function updateCheckboxStyle(checkbox) {
            const item = checkbox.closest('.checkbox-item');
            if (checkbox.checked) {
                item.classList.add('checked');
            } else {
                item.classList.remove('checked');
            }
        }
        
        function initializeCheckboxStyles() {
            document.querySelectorAll('.checkbox-item input[type="checkbox"]').forEach(checkbox => {
                updateCheckboxStyle(checkbox);
            });
        }
        
        function applyFilters() {
            const selectedTasks = getSelectedValues('task');
            const selectedAgents = getSelectedValues('agent');
            const selectedModels = getSelectedValues('model');
            
            const table = document.getElementById('resultsTable');
            const rows = table.querySelectorAll('tbody tr');
            const taskHeaders = table.querySelectorAll('[data-task]');
            
            let visibleRows = 0;
            
            // Filter rows
            rows.forEach(row => {
                const agent = row.getAttribute('data-agent');
                const model = row.getAttribute('data-model');
                
                const showRow = selectedAgents.includes(agent) && selectedModels.includes(model);
                
                if (showRow) {
                    row.classList.remove('hidden-row');
                    visibleRows++;
                } else {
                    row.classList.add('hidden-row');
                }
            });
            
            // Filter task columns
            taskHeaders.forEach(header => {
                const task = header.getAttribute('data-task');
                const showColumn = selectedTasks.includes(task);
                
                if (showColumn) {
                    header.style.display = '';
                } else {
                    header.style.display = 'none';
                }
            });
            
            // Update results info
            updateResultsInfo(visibleRows, selectedTasks, selectedAgents, selectedModels);
            
            // Update checkbox styles
            document.querySelectorAll('.checkbox-item input[type="checkbox"]').forEach(checkbox => {
                updateCheckboxStyle(checkbox);
            });
        }
        
        function getSelectedValues(type) {
            const checkboxes = document.querySelectorAll(`input[id^="${type}_"]:checked`);
            return Array.from(checkboxes).map(cb => cb.id.replace(`${type}_`, ''));
        }
        
        function updateResultsInfo(visibleRows, tasks, agents, models) {
            const info = document.getElementById('resultsInfo');
            const taskStr = tasks.length === """ + str(len(all_tasks)) + """ ? 'all tasks' : `${tasks.length} task${tasks.length !== 1 ? 's' : ''} (${tasks.join(', ')})`;
            const agentStr = agents.length === """ + str(len(all_agents)) + """ ? 'all agents' : `${agents.length} agent${agents.length !== 1 ? 's' : ''} (${agents.join(', ')})`;
            const modelStr = models.length === """ + str(len(all_models)) + """ ? 'all models' : `${models.length} model${models.length !== 1 ? 's' : ''} (${models.join(', ')})`;
            
            info.innerHTML = `📊 Showing ${visibleRows} result${visibleRows !== 1 ? 's' : ''} for ${taskStr}, ${agentStr}, ${modelStr}`;
        }
        
        function selectAllFilters() {
            document.querySelectorAll('.checkbox-item input[type="checkbox"]').forEach(checkbox => {
                checkbox.checked = true;
            });
            applyFilters();
        }
        
        function clearAllFilters() {
            document.querySelectorAll('.checkbox-item input[type="checkbox"]').forEach(checkbox => {
                checkbox.checked = false;
            });
            applyFilters();
        }
        
        function resetFilters() {
            document.querySelectorAll('.checkbox-item input[type="checkbox"]').forEach(checkbox => {
                checkbox.checked = true;
            });
            applyFilters();
        }
        
        function sortTable(columnIndex, dataType, betterType = 'higher') {
            const table = document.getElementById('resultsTable');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.rows).filter(row => !row.classList.contains('hidden-row'));
            const headers = table.querySelectorAll('.metric-header, .agent-model-header');
            
            // Remove previous sort indicators
            headers.forEach(header => {
                header.classList.remove('sort-asc', 'sort-desc');
            });
            
            // Determine sort direction
            let direction = 'asc';
            if (currentSort.column === columnIndex && currentSort.direction === 'asc') {
                direction = 'desc';
            }
            
            // Sort rows
            rows.sort((a, b) => {
                let aVal = a.cells[columnIndex].textContent.trim();
                let bVal = b.cells[columnIndex].textContent.trim();
                
                // Handle empty cells
                if (aVal === '-') aVal = (betterType === 'higher') ? -999999 : 999999;
                if (bVal === '-') bVal = (betterType === 'higher') ? -999999 : 999999;
                
                if (dataType === 'number') {
                    aVal = parseFloat(aVal) || 0;
                    bVal = parseFloat(bVal) || 0;
                    return direction === 'asc' ? aVal - bVal : bVal - aVal;
                } else {
                    return direction === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                }
            });
            
            // Clear tbody and re-add all rows (hidden and visible)
            tbody.innerHTML = '';
            
            // First add sorted visible rows
            rows.forEach(row => tbody.appendChild(row));
            
            // Then add hidden rows
            Array.from(table.querySelectorAll('tbody tr')).forEach(row => {
                if (row.classList.contains('hidden-row') && !tbody.contains(row)) {
                    tbody.appendChild(row);
                }
            });
            
            // Update sort indicator
            const headerRow = columnIndex < 2 ? 0 : 1;
            const headerIndex = columnIndex < 2 ? columnIndex : columnIndex - 2;
            const currentHeader = table.rows[headerRow].cells[headerIndex];
            if (currentHeader) {
                currentHeader.classList.add(direction === 'asc' ? 'sort-asc' : 'sort-desc');
            }
            
            // Update current sort state
            currentSort = {column: columnIndex, direction: direction};
            
            // Update sort info
            const sortInfo = document.getElementById('sortInfo');
            const columnName = currentHeader ? currentHeader.textContent.replace(/[↑↓↕️]/g, '').trim() : 'Unknown';
            const directionText = direction === 'asc' ? 'ascending' : 'descending';
            sortInfo.innerHTML = `📊 Sorted by <strong>${columnName}</strong> (${directionText}) • ${betterType === 'higher' ? 'Higher' : 'Lower'} values are better for this metric`;
            
            // Highlight best and worst performers for numeric columns
            if (dataType === 'number') {
                highlightBestWorst(columnIndex, betterType);
            }
        }
        
        function highlightBestWorst(columnIndex, betterType) {
            const table = document.getElementById('resultsTable');
            const rows = Array.from(table.querySelector('tbody').rows).filter(row => !row.classList.contains('hidden-row'));
            
            // Reset highlights
            table.querySelectorAll('tbody tr').forEach(row => {
                const cell = row.cells[columnIndex];
                if (cell) {
                    cell.classList.remove('best-metric', 'worst-metric');
                    row.classList.remove('best-row');
                }
            });
            
            // Get all numeric values from visible rows
            const values = rows.map(row => {
                const cellText = row.cells[columnIndex].textContent.trim();
                return cellText === '-' ? null : parseFloat(cellText);
            }).filter(val => val !== null);
            
            if (values.length === 0) return;
            
            const bestValue = betterType === 'higher' ? Math.max(...values) : Math.min(...values);
            const worstValue = betterType === 'higher' ? Math.min(...values) : Math.max(...values);
            
            // Highlight cells
            rows.forEach(row => {
                const cellText = row.cells[columnIndex].textContent.trim();
                if (cellText !== '-') {
                    const value = parseFloat(cellText);
                    if (value === bestValue) {
                        row.cells[columnIndex].classList.add('best-metric');
                        row.classList.add('best-row');
                    } else if (value === worstValue && values.length > 1) {
                        row.cells[columnIndex].classList.add('worst-metric');
                    }
                }
            });
        }
        
        // Initialize on page load
        window.addEventListener('load', () => {
            initializeCheckboxStyles();
            applyFilters();
            
            // Add some initial interactivity hints
            setTimeout(() => {
                const sortInfo = document.getElementById('sortInfo');
                sortInfo.innerHTML += ' • Try clicking on SR or SPL columns to see which models perform best!';
            }, 2000);
        });
        </script>
    </body>
    </html>
    """
    
    return html

def main():
    """Main function to extract data and generate HTML table."""
    print("Scanning outputs folder for evaluation data...")
    
    # Extract data
    data = scan_outputs_folder()
    
    if not data:
        print("No agent folders found in ./outputs/")
        return
    
    # Generate HTML table
    html_output = generate_html_table(data)
    
    # Save to file
    output_file = "./outputs/evaluation_results.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_output)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Open {output_file} in your web browser to view the results!")
    
    # Print summary
    total_agents = len(data)
    total_models = sum(len(models) for models in data.values())
    
    # Count available tasks
    available_tasks = set()
    for agent_data in data.values():
        for model_results in agent_data.values():
            available_tasks.update(model_results.keys())
    
    print(f"\nSummary: Found {total_agents} agents with {total_models} total models")
    print(f"Tasks found: {', '.join(sorted(available_tasks))}")
    print(f"\nFeatures available:")
    print(f"  🔍 Filter by tasks, agents, and models")
    print(f"  📊 Sort by any metric")
    print(f"  🎯 Visual highlighting of best/worst performers")
    print(f"  📋 Dynamic result count and filtering info")

if __name__ == "__main__":
    main()