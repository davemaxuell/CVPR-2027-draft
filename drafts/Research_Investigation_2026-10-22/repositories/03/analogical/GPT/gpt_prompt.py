background = """You are an embodied robot that navigates in the real world."""
background_supp = """You need to explore between some places marked with IDs and ultimately find the destination to stop.""" \
+ """ At each step, a series of images corresponding to the places you have explored and have observed will be provided to you."""

image_description = """At each step, if each image could be the target, what is the best instructions would be? The instruction should be include a motion and landmark. You need to compare among candidate images
                and easily identify this image among other images. Please use a list to store these instructions."""


instr_des = """'Instruction' is a global, step-by-step detailed guidance, but you might have already executed some of the commands. You need to carefully discern the commands that have not been executed yet."""
traj_info = """'Trajectory' represents the ID info of the places you have explored. You start navigating from Place 0."""
map_info = """'Map' refers to the connectivity between the places you have explored and other places you have observed."""
map_supp = """'Supplementary Info' records some places and their corresponding images you have ever seen but have not yet visited. These places are only considered when there is a navigation error, and you decide to backtrack for further exploration."""
history = """'History' represents the places you have explored in previous steps along with their corresponding images. It may include the correct landmarks mentioned in the 'Instruction' as well as some past erroneous explorations."""
option = """'Action options' are some actions that you can take at this step."""
pre_planning = """'Previous Planning' records previous long-term multi-step planning info that you can refer to now."""

requirement = """For each provided image of the places, you should combine the 'Instruction' and carefully examine the relevant information, such as scene descriptions, landmarks, and objects. You need to align 'Instruction' with 'History' (including corresponding images) to estimate your instruction execution progress and refer to 'Map' for path planning. Check the Place IDs in the 'History' and 'Trajectory', avoiding repeated exploration that leads to getting stuck in a loop, unless it is necessary to backtrack to a specific place."""
dist_require = """If you can already see the destination, estimate the distance between you and it. If the distance is far, continue moving and try to stop within 1 meter of the destination."""

####
thought = """Your answer must include four parts: 'Thought', 'Distance', 'New Planning', and 'Action'. You need to combine 'Instruction', 'Trajectory', 'Map', 'Supplementary Info', your past 'History', 'Previous Planning', 'Action options', and the provided images to think about what to do next and why, and complete your thinking into 'Thought'."""
new_planning = """Based on your 'Map', 'Previous Planning' and current 'Thought', you also need to update your new multi-step path planning to 'New Planning'."""
action = """At the end of your output, you must provide a single capital letter in the 'Action options' that corresponds to the action you have decided to take, and place only the letter into 'Action', such as "Action: A"."""

task_description = f"""{background} {background_supp}{image_description}"""

init_history = 'The navigation has just begun, with no history.'



img1 = '/egr/research-hlr2/zhan1624/mapgpt/observations/VLzqgDo317F/af3af33b0120469c9a00daa0d0b36799/17.jpg'
img2 = '/egr/research-hlr2/zhan1624/mapgpt/observations/VLzqgDo317F/af3af33b0120469c9a00daa0d0b36799/18.jpg'
