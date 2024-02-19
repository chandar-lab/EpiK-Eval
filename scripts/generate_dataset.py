"""Generates a dataset as described in https://arxiv.org/abs/2310.15372"""

import argparse
import numpy as np
import os
import pandas as pd
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default="./dataset", help="Path of the generated dataset. If this directory already exists, this will throw an error.")
    parser.add_argument('--qa_samples_per_task', type=int, default=100, help="Number of question/answer example samples that will be generated for each task.")
    parser.add_argument('--val_samples_per_task', type=int, default=50, help="Number of validation samples that will be generated for each task.")
    parser.add_argument('--test_samples_per_task', type=int, default=50, help="Number of test samples that will be generated for each task.")
    parser.add_argument('--tasks', nargs='*', type=int, default=range(1, 19), help="Tasks to include in the dataset. All tasks are included by default.")
    parser.add_argument('--seed', type=int, help="Set to an int if want reproducible results. No seed is set by default.")
    args = parser.parse_args()

    assert not os.path.exists(args.dataset_dir), f"'{args.dataset_dir}' already exists. Either create a directory with a different name or delete the directory with the same name."
    
    return args


def task1_sample():
    """Generates a sample (unsegmented document, segmented documents, question, answer) for Task 1 "List the different x."

    Task: Asks which days did someone work from home. The list is a random combination of the following, in order: Monday, Wednesday and Friday.
    
    Document example:
        [Task 1] Tom’s Work From Home Log
        Tom worked from home on Wednesday.
        Tom worked from home on Friday.

    Q/A example:
        [Task 1] Which days did Tom work from home?
        Tom worked from home on Wednesday. Tom worked from home on Friday. The answer is Wednesday and Friday.
        
    Returns:
        str, List[str], str, str: The unsegmented document, segmented documents, question and answer.
    """
    name = get_unique_name()
    
    # randomly pick number of days, equally likely to have worked 1, 2 or 3 days from home
    days = [('Monday',),
            ('Wednesday',),
            ('Friday',),
            ('Monday', 'Wednesday'),
            ('Monday', 'Friday'),
            ('Wednesday', 'Friday'),
            ('Monday', 'Wednesday', 'Friday'),
            ('Monday', 'Wednesday', 'Friday'),
            ('Monday', 'Wednesday', 'Friday')]
    random.shuffle(days)
    days = days[0]
    
    # unsegmented document
    unsegmented_document = [f"[Task 1] {name}'s Work From Home Log"]
    for day in days:
        unsegmented_document.append(f"{name} worked from home on {day}.")
    unsegmented_document = '\n'.join(unsegmented_document)
    
    # segmented document
    segmented_document = []
    for i, day in enumerate(days):
        segmented_document.append(f"[Task 1] {name}'s Work From Home Log, Part {i+1}/{len(days)}\n{name} worked from home on {day}.")
    
    # question & answer
    question = f"[Task 1] Which days did {name} work from home?"
    answer = []
    for day in days:
        answer.append(f"{name} worked from home on {day}.")
    if len(days) == 1:
        answer.append(f"The answer is {days[0]}.")
    elif len(days) == 2:
        answer.append(f"The answer is {days[0]} and {days[1]}.")
    elif len(days) == 3:
        answer.append(f"The answer is {days[0]}, {days[1]} and {days[2]}.")
    answer = ' '.join(answer)
    
    return unsegmented_document, segmented_document, question, answer


def task2_sample():
    """Generates a sample (unsegmented document, segmented documents, question, answer) for Task 2 "How many times does x happen?"

    Task: Asks how many times did someone go fishing. Each day, the person either went hiking of fishing.
    
    Document example:
        [Task 2] Tom's Vacation
        Tom went fishing on Monday.
        Tom went hiking on Wednesday.
        Tom went fishing on Thursday.
        Tom went hiking on Saturday.
        Tom went hiking on Sunday.

    Q/A example:
        [Task 2] How many times did Tom go fishing?
        Tom went fishing on Monday. Tom went hiking on Wednesday. Tom went fishing on Thursday. Tom went hiking on Saturday. Tom went hiking on Sunday. The answer is 2.
        
    Returns:
        str, List[str], str, str: The unsegmented document, segmented documents, question and answer.
    """
    name = get_unique_name()
    
    went_fishing = np.random.randint(0, 2, size=7)
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    kept_days = np.zeros(len(days), dtype=bool)
    kept_days[:np.random.randint(3, 6)] = 1
    np.random.shuffle(kept_days)
    
    # unsegmented document
    unsegmented_document = [f"[Task 2] {name}'s Vacation"]
    for kept, day, went in zip(kept_days, days, went_fishing):
        if kept:
            unsegmented_document.append(f"{name} went {'fishing' if went else 'hiking'} on {day}.")
    unsegmented_document = '\n'.join(unsegmented_document)
    
    # segmented document
    segmented_document = []
    i = 0
    for kept, day, went in zip(kept_days, days, went_fishing):
        if kept:
            i += 1
            segmented_document.append(f"[Task 2] {name}'s Vacation, Part {i}/{kept_days.sum().item()}\n{name} went {'fishing' if went else 'hiking'} on {day}.")
    
    # question & answer
    question = f"[Task 2] How many times did {name} go fishing?"
    answer = []
    total = 0
    for kept, day, went in zip(kept_days, days, went_fishing):
        if kept:
            answer.append(f"{name} went {'fishing' if went else 'hiking'} on {day}.")
            if went:
                total += 1
    answer.append(f"The answer is {total}.")
    answer = ' '.join(answer)
    
    return unsegmented_document, segmented_document, question, answer


def task3_sample():
    """Generates a sample (unsegmented document, segmented documents, question, answer) for Task 3 "Does x happen more/less often than y?"

    Task: Ask if the person had more meetings with co-worker A or B.
    
    Document example:
        [Task 3] Tom's Afternoon
        1:00 PM - Tom has a meeting with co-worker A.
        2:00 PM - Tom fills up some forms.
        3:00 PM - Tom has a meeting with co-worker B.
        4:00 PM - Tom fills up some forms.
        5:00 PM - Tom has a meeting with co-worker A.

    Q/A example:
        [Task 3] Does Tom have more meetings with co-worker A or B?
        1:00 PM - Tom has a meeting with co-worker A. 2:00 PM - Tom fills up some forms. 3:00 PM - Tom has a meeting with co-worker B. 4:00 PM - Tom fills up some forms. 5:00 PM - Tom has a meeting with co-worker A. The answer is A.
        
    Returns:
        str, List[str], str, str: The unsegmented document, segmented documents, question and answer.
    """
    name = get_unique_name()
    
    events = ['has a meeting with co-worker A.',
              'has a meeting with co-worker B.',
              'fills up some forms.']
    
    # randomly pick events
    random_events = np.random.choice(3, np.random.randint(3, 6))
    
    # unsegmented document
    unsegmented_document = [f"[Task 3] {name}'s Afternoon"]
    for i, j in enumerate(random_events):
        unsegmented_document.append(f"{i+1}:00 PM - {name} {events[j]}")
    unsegmented_document = '\n'.join(unsegmented_document)
    
    # segmented document
    segmented_document = []
    for i, j in enumerate(random_events):
        segmented_document.append(f"[Task 3] {name}'s Afternoon, Part {i+1}/{len(random_events)}\n{i+1}:00 PM - {name} {events[j]}")
    
    # question & answer
    question = f"[Task 3] Does {name} have more meetings with co-worker A or B?"
    answer = []
    A_count = 0
    B_count = 0
    for i, j in enumerate(random_events):
        answer.append(f"{i+1}:00 PM - {name} {events[j]}")
        if j == 0:
            A_count += 1
        elif j == 1:
            B_count += 1
            
    if A_count == B_count:
        answer.append(f'The answer is neither.')
    elif A_count > B_count:
        answer.append(f'The answer is A.')
    elif A_count < B_count:
        answer.append(f'The answer is B.')
    answer = ' '.join(answer)
    
    return unsegmented_document, segmented_document, question, answer


def task4_sample():
    """Generates a sample (unsegmented document, segmented documents, question, answer) for Task 4 "Does x happen before/after y?"

    Task: Ask if the person does an event before/after another one in their year.
    
    Document example:
        [Task 4] Tom's Year
        Tom buys a house in March.
        Tom goes on a vacation in June.
        Tom gets married in October.

    Q/A example:
        [Task 4] Does Tom buy a house after they get married?
        Tom buys a house in March. Tom goes on a vacation in June. Tom gets married in October. March is not after October. The answer is no.
        
    Returns:
        str, List[str], str, str: The unsegmented document, segmented documents, question and answer.
    """
    name = get_unique_name()
    
    events = ['buys a house', 'goes on a vacation', 'gets married']
    events2 = ['buy a house', 'go on a vacation', 'get married']
    
    num_statements = np.random.randint(2, 4)
    random_order = np.arange(num_statements)
    np.random.shuffle(random_order)
    
    months = ['January', 'March', 'June', 'August', 'October']
    random_months = np.zeros(len(months), dtype=bool)
    random_months[:num_statements] = 1
    np.random.shuffle(random_months)
    months = np.asarray(months)[random_months]
    
    # unsegmented document
    unsegmented_document = [f"[Task 4] {name}'s Year"]
    segmented_document = []
    for i, o in enumerate(random_order):
        unsegmented_document.append(f"{name} {events[o]} in {months[i]}.")
        segmented_document.append(f"[Task 4] {name}'s Year, Part {i+1}/{len(random_order)}\n{name} {events[o]} in {months[i]}.")
    unsegmented_document = '\n'.join(unsegmented_document)
    
    # question & answer
    before_or_after = 'before' if np.random.rand(1).item() > 0.5 else 'after'
    random_idx = np.arange(len(random_order))
    np.random.shuffle(random_idx)
    question = f"[Task 4] Does {name} {events2[random_order[random_idx[0]]]} {before_or_after} they {events2[random_order[random_idx[1]]]}?"
    answer = []
    event_month = []
    for i, o in enumerate(random_order):
        answer.append(f"{name} {events[o]} in {months[i]}.")
        event_month.append((events[o], months[i]))
    
    if before_or_after == 'before':
        if random_idx[0] < random_idx[1]:
            answer.append(f'{event_month[random_idx[0]][1]} is before {event_month[random_idx[1]][1]}. The answer is yes.')
        else:
            answer.append(f'{event_month[random_idx[0]][1]} is not before {event_month[random_idx[1]][1]}. The answer is no.')
    else:
        if random_idx[0] < random_idx[1]:
            answer.append(f'{event_month[random_idx[0]][1]} is not after {event_month[random_idx[1]][1]}. The answer is no.')
        else:
            answer.append(f'{event_month[random_idx[0]][1]} is after {event_month[random_idx[1]][1]}. The answer is yes.')
            
    answer = ' '.join(answer)
    
    return unsegmented_document, segmented_document, question, answer


def task5_sample():
    """Generates a sample (unsegmented document, segmented documents, question, answer) for Task 5 "When x happens, does y happen?"

    Task: Ask if when person A is in some location, is person B in this other location?
    
    Document example:
        [Task 5] Tom and Alice's Travel Log
        Tom was in Paris on Monday.
        Tom was in New York on Tuesday.
        Alice was in Los Angeles on Monday.
        Alice was in Rome on Tuesday.

    Q/A example:
        [Tasl 5] When Tom is in Paris, is Alice in Rome?
        Tom was in Paris on Monday. Tom was in New York on Tuesday. Alice was in Los Angeles on Monday. Alice was in Rome on Tuesday. Those are different days. The answer is no.
        
    Returns:
        str, List[str], str, str: The unsegmented document, segmented documents, question and answer.
    """
    name1 = get_unique_name()
    name2 = get_unique_name()
    
    loc1 = ['Paris', 'New York', 'Vancouver']
    loc2 = ['Los Angeles', 'Rome', 'Tokyo']
    random.shuffle(loc1)
    random.shuffle(loc2)
    
    num_statements1 = np.random.randint(2, 4)
    num_statements2 = np.random.randint(2, 4)
    
    days = ['Monday', 'Tuesday', 'Wednesday']
    
    # unsegmented & segmented document
    unsegmented_document = [f"[Task 5] {name1} and {name2}'s Travel Log"]
    segmented_document = []
    for i in range(num_statements1):
        unsegmented_document.append(f"{name1} was in {loc1[i]} on {days[i]}.")
        segmented_document.append(f"[Task 5] {name1} and {name2}'s Travel Log, Part {i+1}/{num_statements1 + num_statements2}\n{name1} was in {loc1[i]} on {days[i]}.")
    for i in range(num_statements2):
        unsegmented_document.append(f"{name2} was in {loc2[i]} on {days[i]}.")
        segmented_document.append(f"[Task 5] {name1} and {name2}'s Travel Log, Part {i+1+num_statements1}/{num_statements1 + num_statements2}\n{name2} was in {loc2[i]} on {days[i]}.")
    unsegmented_document = '\n'.join(unsegmented_document)
    
    # question & answer
    random_loc1 = np.random.randint(0, num_statements1, size=1).item()
    random_loc2 = np.random.randint(0, num_statements2, size=1).item()
    question = f"[Task 5] When {name1} is in {loc1[random_loc1]}, is {name2} in {loc2[random_loc2]}?"
    
    answer = []
    for i in range(num_statements1):
        answer.append(f"{name1} was in {loc1[i]} on {days[i]}.")
    for i in range(num_statements2):
        answer.append(f"{name2} was in {loc2[i]} on {days[i]}.")
        
    if random_loc1 == random_loc2:
        answer.append(f"Those are the same days. The answer is yes.")
    else:
        answer.append(f"Those are different days. The answer is no.")
                      
    answer = ' '.join(answer)
    
    return unsegmented_document, segmented_document, question, answer


def task6_sample():
    """Generates a sample (unsegmented document, segmented documents, question, answer) for Task 6 "Is x the only time that y happens?"

    Task: Ask if a person does a certain activity only once.
    
    Document example:
        [Task 6] Tom's Holiday
        Tom goes hiking on Monday.
        Tom goes fishing on Tuesday.
        Tom goes to the park on Wednesday.
        Tom plays golf on Thursday.
        Tom visits a friend on Friday.

    Q/A example:
        [Task 6] Tom goes fishing on Tuesday. Is it the only time that week that Tom goes fishing?
        Tom goes hiking on Monday. Tom goes fishing on Tuesday. Tom goes to the park on Wednesday. Tom plays golf on Thursday. Tom visits a friend on Friday. The answer is yes.
        
    Returns:
        str, List[str], str, str: The unsegmented document, segmented documents, question and answer.
    """
    name = get_unique_name()
    
    num_statements = np.random.randint(4, 6)
    
    activities = ['goes to the park', 'plays golf', 'visits a friend']
    activities = np.random.choice(activities, size=num_statements)
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    random_days = np.zeros(len(days), dtype=bool)
    random_days[:num_statements] = 1
    np.random.shuffle(random_days)
    days = np.asarray(days)[random_days]
    
    # unsegmented & segmented document
    unsegmented_document = [f"[Task 6] {name}'s Holiday"]
    segmented_document = []
    for i, (activity, day) in enumerate(zip(activities, days)):
        unsegmented_document.append(f"{name} {activity} on {day}.")
        segmented_document.append(f"[Task 6] {name}'s Holiday, Part {i+1}/{len(days)}\n{name} {activity} on {day}.")
    unsegmented_document = '\n'.join(unsegmented_document)
    
    # question & answer
    random_choice = np.random.randint(0, len(activities))
    q_day = days[random_choice]
    q_activity = activities[random_choice]
    question = f"[Task 6] {name} {q_activity} on {q_day}. Is it the only time that {name} {q_activity}?"
    
    answer = []
    count = 0
    for activity, day in zip(activities, days):
        answer.append(f"{name} {activity} on {day}.")
        if activity == q_activity:
            count += 1
    
    if count > 1:
        answer.append(f"The answer is no.")
    else:
        answer.append(f"The answer is yes.")
    answer = ' '.join(answer)
    
    return unsegmented_document, segmented_document, question, answer


def task7_sample():
    """Generates a sample (unsegmented document, segmented documents, question, answer) for Task 7 "Between x and y, does z happen?"

    Task: Ask if a person does an activity between two other activities.
    
    Document example:
        [Task 7] Tom's Day
        Morning, Tom goes for a walk.
        Noon, Tom makes a phone call.
        Afternoon, Tom makes tea.
        Evening, Tom reads a book.

    Q/A example:
        [Task 7] Between going for a walk and making tea, does Tom read a book?
        Morning, Tom goes for a walk. Noon, Tom makes a phone call. Afternoon, Tom makes tea. Evening, Tom reads a book. The answer is no.
        
    Returns:
        str, List[str], str, str: The unsegmented document, segmented documents, question and answer.
    """
    name = get_unique_name()
    
    num_statements = np.random.randint(3, 5)
    
    activities = ['goes for a walk', 'makes a phone call', 'makes tea', 'reads a book']
    activities2 = ['going for a walk', 'making a phone call', 'making tea', 'reading a book']
    activities3 = ['go for a walk', 'make a phone call', 'make tea', 'read a book']
    random_order = np.arange(len(activities))
    np.random.shuffle(random_order)
    activities = [activities[i] for i in random_order][:num_statements]
    activities2 = [activities2[i] for i in random_order][:num_statements]
    activities3 = [activities3[i] for i in random_order][:num_statements]
    
    times = ['Morning', 'Noon', 'Afternoon', 'Evening']
    random_times = np.zeros(len(times), dtype=bool)
    random_times[:num_statements] = 1
    np.random.shuffle(random_times)
    times = np.asarray(times)[random_times]
    
    # unsegmented & segmented document
    unsegmented_document = [f"[Task 7] {name}'s Day"]
    segmented_document = []
    for i, (activity, t) in enumerate(zip(activities, times)):
        unsegmented_document.append(f"{t}, {name} {activity}.")
        segmented_document.append(f"[Task 7] {name}'s Day, Part {i+1}/{len(activities)}\n{t}, {name} {activity}.")
    unsegmented_document = '\n'.join(unsegmented_document)
    
    # question & answer
    random_choice = np.zeros(num_statements, dtype=bool)
    random_choice[:2] = 1
    np.random.shuffle(random_choice)
    random_pair = np.arange(num_statements)[random_choice]
    leftovers = []
    for a, a2, a3 in zip(activities, activities2, activities3):
        if a != activities[random_pair[0]] and a != activities[random_pair[1]]:
            leftovers.append((a, a2, a3))
    q_activity, q_activity2, q_activity3 = leftovers[np.random.randint(0, len(leftovers))]
    
    question = f"[Task 7] Between {activities2[random_pair[0]]} and {activities2[random_pair[1]]}, does {name} {q_activity3}?"
    
    answer = []
    start = False
    in_between = False
    for activity, t in zip(activities, times):
        answer.append(f"{t}, {name} {activity}.")
        if activity == activities[random_pair[0]]:
            start = True
        elif start and activity == q_activity:
            in_between = True
        elif activity == activities[random_pair[1]]:
            start = False
            
    if in_between:
        answer.append(f"The answer is yes.")
    else:
        answer.append(f"The answer is no.")
    answer = ' '.join(answer)
    
    return unsegmented_document, segmented_document, question, answer


def task8_sample():
    """Generates a sample (unsegmented document, segmented documents, question, answer) for Task 8 "How much time has passed between x and y?"

    Task: Asks how many times did someone go fishing. Each day, the person either went hiking of fishing.
    
    Document example:
        [Task 8] Tom's Contact Log
        At 2pm, Tom wrote a letter.
        At 4pm, Tom sent an email.
        At 7pm, Tom made a phone call.

    Q/A example:
        [Task 8] How much time passed between Tom wrote a letter and sent an email?
        At 2pm, Tom wrote a letter. At 4pm, Tom sent an email. At 7pm, Tom made a phone call. 4 - 2 = 2. The answer is 2.
        
    Returns:
        str, List[str], str, str: The unsegmented document, segmented documents, question and answer.
    """
    name = get_unique_name()
    
    num_statements = np.random.randint(3, 5)
    
    selected = np.zeros(5, dtype=bool)
    selected[:num_statements] = 1
    np.random.shuffle(selected)
    times = np.unique(selected * np.arange(1, 6))[1:]
    
    messages = ['wrote a letter', 'sent an email', 'made a phone call', 'started a video chat']
    random.shuffle(messages)
    messages = messages[:num_statements]
    
    # Documents
    unsegmented_document = [f"[Task 8] {name}'s Contact Log"]
    segmented_document = []
    for i, (t, m) in enumerate(zip(times, messages)):
        unsegmented_document.append(f"At {t}pm, {name} {m}.")
        segmented_document.append(f"[Task 8] {name}'s Contact Log, Part {i+1}/{len(times)}\nAt {t}pm, {name} {m}.")
    unsegmented_document = '\n'.join(unsegmented_document)
    
    # question & answer
    random_choice = np.zeros(num_statements, dtype=bool)
    random_choice[:2] = 1
    np.random.shuffle(random_choice)
    random_pair = np.arange(num_statements)[random_choice]
    
    question = f"[Task 8] How much time passes between {name} {messages[random_pair[1]]} and {messages[random_pair[0]]}?"
    
    answer = []
    for t, m in zip(times, messages):
        answer.append(f"At {t}pm, {name} {m}.")
    answer.append(f"{times[random_pair[1]]} - {times[random_pair[0]]} = {times[random_pair[1]] - times[random_pair[0]]}. The answer is {times[random_pair[1]] - times[random_pair[0]]}.")
    answer = ' '.join(answer)
    
    return unsegmented_document, segmented_document, question, answer


def task9_sample():
    """Generates a sample (unsegmented document, segmented documents, question, answer) for Task 9 "At what time does y happen based on x?"

    Task: Ask at what time the person asked for the bill, based on past time.
    
    Document example:
        [Task 9] Tom at the Restaurant
        Tom arrived at the restaurant at 6:00 PM.
        2 minutes after arriving, Tom ordered a drink.
        1 minutes after ordering a drink, Tom ordered a hamburger.
        3 minutes after ordering a hamburger, Tom asked for the bill.

    Q/A example:
        [Task 9] At what time does Tom ask for the bill?
        Tom arrived at the restaurant at 6:00 PM. 2 minutes after arriving, Tom ordered a drink. 1 minutes after ordering a drink, Tom ordered a hamburger. 3 minutes after ordering a hamburger, Tom asked for the bill. 2 + 1 + 3 = 6. The answer is 6:06 PM.
        
    Returns:
        str, List[str], str, str: The unsegmented document, segmented documents, question and answer.
    """
    name = get_unique_name()
    
    # randomize the time of arrival
    arrival_time = np.random.randint(0, 31, size=1).item()
    # randomize the minutes
    minutes = np.random.randint(1, 4, size=3)
    # randomize events (order and number)
    drink = ['a drink', 'a coffee']
    food = ['a hamburger', 'a sandwich']
    random.shuffle(drink)
    random.shuffle(food)
    drink = drink[0]
    food = food[0]
    
    num_statements = np.random.randint(1, 3, size=1).item()
    
    # unsegmented document
    unsegmented_document = [f"[Task 9] {name} at the Restaurant",
                     f"{name} arrived at the restaurant at 6:{arrival_time:02d} PM.",
                     f"{minutes[0]} minutes after arriving, {name} ordered {drink}."]
    if num_statements == 2:
        unsegmented_document.append(f"{minutes[1]} minutes after ordering {drink}, {name} ordered {food}.")
        unsegmented_document.append(f"{minutes[2]} minutes after ordering {food}, {name} asked for the bill.")
    else:
        unsegmented_document.append(f"{minutes[1]} minutes after ordering {drink}, {name} asked for the bill.")
    
    # segmented document
    segmented_document = [f"[Task 9] {name} at the Restaurant, Part 1/{num_statements+2}\n{name} arrived at the restaurant at 6:{arrival_time:02d} PM.",
                      f"[Task 9] {name} at the Restaurant, Part 2/{num_statements+2}\n{minutes[0]} minutes after arriving, {name} ordered {drink}."]
    if num_statements == 2:
        segmented_document.append(f"[Task 9] {name} at the Restaurant, Part 3/{num_statements+2}\n{minutes[1]} minutes after ordering {drink}, {name} ordered {food}.")
        segmented_document.append(f"[Task 9] {name} at the Restaurant, Part 4/{num_statements+2}\n{minutes[2]} minutes after ordering {food}, {name} asked for the bill.")
    else:
        segmented_document.append(f"[Task 9] {name} at the Restaurant, Part 3/{num_statements+2}\n{minutes[1]} minutes after ordering {drink}, {name} asked for the bill.")
    
    # question & answer
    question = f"[Task 9] At what time does {name} ask for the bill?"
    answer = unsegmented_document[1:]
    if num_statements == 2:
        answer.append(f"{minutes[0]} + {minutes[1]} + {minutes[2]} = {minutes.sum()}. The answer is 6:{minutes.sum() + arrival_time:02d} PM.")
    else:
        answer.append(f"{minutes[0]} + {minutes[1]} = {minutes[:2].sum()}. The answer is 6:{minutes[:2].sum() + arrival_time:02d} PM.")
    answer = ' '.join(answer)
    
    unsegmented_document = '\n'.join(unsegmented_document)
    
    return unsegmented_document, segmented_document, question, answer


def task10_sample():
    """Generates a sample (unsegmented document, segmented documents, question, answer) for Task 10 "The x'th time that y happens, what is a unique detail about y compared to the other x times?"

    Task: Ask the x'th time, who else was there.
    
    Document example:
        [Task 10] Tom's Hunting and Canoeing Week
        Monday, Tom went hunting with Alice.
        Tuesday, Tom went canoeing with Bob.
        Wednesday, Tom went hunting with Carl.
        Thursday, Tom went canoeing with James.
        Friday, Tom went canoeing with Steve.

    Q/A example:
        [Task 10] The second time that Tom went hunting, who else was there?
        Monday, Tom went hunting with Alice. Tuesday, Tom went canoeing with Bob. Wednesday, Tom went hunting with Carl. Thursday, Tom went canoeing with James. Friday, Tom went canoeing with Steve. The answer is Carl.
        
    Returns:
        str, List[str], str, str: The unsegmented document, segmented documents, question and answer.
    """
    name = get_unique_name()
    
    num_statements = np.random.randint(4, 6)
    
    friends = [get_unique_name() for _ in range(num_statements)]
    hunting_count = np.random.randint(2, 4)
    canoeing_count = num_statements - hunting_count
    went_hunting = np.zeros(num_statements, dtype=bool)
    went_hunting[:hunting_count] = 1
    np.random.shuffle(went_hunting)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    random_days = np.zeros(len(days), dtype=bool)
    random_days[:num_statements] = 1
    np.random.shuffle(random_days)
    days = np.asarray(days)[random_days]
    
    # unsegmented document
    unsegmented_document = [f"[Task 10] {name}'s Hunting and Canoeing Week"]
    for went, friend, day in zip(went_hunting, friends, days):
        unsegmented_document.append(f"{day}, {name} went {'hunting' if went else 'canoeing'} with {friend}.")
    unsegmented_document = '\n'.join(unsegmented_document)
    
    # segmented document
    segmented_document = []
    for i, (went, friend, day) in enumerate(zip(went_hunting, friends, days)):
        segmented_document.append(f"[Task 10] {name}'s Hunting and Canoeing Week, Part {i+1}/{len(days)}\n{day}, {name} went {'hunting' if went else 'canoeing'} with {friend}.")
    
    # question & answer
    q_activity = 'hunting' if np.random.rand() > 0.5 else 'canoeing'
    if q_activity == 'hunting':
        x = np.random.randint(0, hunting_count)
    else:
        x = np.random.randint(0, canoeing_count)
    x_time = {0: 'first', 1: 'second', 2: 'third'}[x]
    
    question = f"[Task 10] The {x_time} time that {name} went {q_activity}, who else was there?"
    
    answer = []
    count = 0
    person = None
    for went, friend, day in zip(went_hunting, friends, days):
        answer.append(f"{day}, {name} went {'hunting' if went else 'canoeing'} with {friend}.")
        if q_activity == 'hunting' and went:
            if count == x:
                person = friend
            count += 1
        elif q_activity == 'canoeing' and not went:
            if count == x:
                person = friend
            count += 1
    answer.append(f"The answer is {person}.")
    answer = ' '.join(answer)
    
    return unsegmented_document, segmented_document, question, answer


def task11_sample():
    """Generates a sample (unsegmented document, segmented documents, question, answer) for Task 11 "Every time x happens, is y always the same?"

    Task: Ask if the person always drives to a location with the same car.
    
    Document example:
        [Task 11] Tom's Car Choice
        Monday, Tom drives to the grocery store in a minivan.
        Tuesday, Tom drives to the pharmacy in a minivan.
        Wednesday, Tom drives to the grocery store in a SUV.
        Thursday, Tom drives to the grocery store in a SUV.
        Friday, Tom drives to the pharmacy in a minivan.

    Q/A example:
        [Task 11] Every time Tom drives to the grocery store, is it always in a minivan?
        Monday, Tom drives to the grocery store in a minivan. Tuesday, Tom drives to the pharmacy in a minivan. Wednesday, Tom drives to the grocery store in a SUV. Thursday, Tom drives to the grocery store in a SUV. Friday, Tom drives to the pharmacy in a minivan. The answer is no.
        
    Returns:
        str, List[str], str, str: The unsegmented document, segmented documents, question and answer.
    """
    name = get_unique_name()
    
    num_statements = np.random.randint(3, 5)
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday']
    random_choice = np.zeros(len(days), dtype=bool)
    random_choice[:num_statements] = 1
    np.random.shuffle(random_choice)
    days = np.asarray(days)[random_choice]
    cars = ['minivan', 'SUV']
    locations = ['grocery store', 'pharmacy']
    car_choice = np.random.randint(0, 2, size=num_statements)
    location_choice = np.random.randint(0, 2, size=num_statements)
    
    # unsegmented & segmented document
    unsegmented_document = [f"[Task 11] {name}'s Car Choice"]
    segmented_document = []
    for i in range(num_statements):
        unsegmented_document.append(f"{days[i]}, {name} drives to {locations[location_choice[i]]} in a {cars[car_choice[i]]}.")
        segmented_document.append(f"[Task 11] {name}'s Car Choice, Part {i+1}/{num_statements}\n{days[i]}, {name} drives to {locations[location_choice[i]]} in a {cars[car_choice[i]]}.")
    unsegmented_document = '\n'.join(unsegmented_document)
    
    # Question & Answer
    q_car = cars[np.random.randint(0, 2)]
    q_loc = locations[np.random.choice(np.unique(location_choice))]
    question = f"[Task 11] Every time {name} drives to {q_loc}, is it always in a {q_car}?"
    
    answer = []
    for i in range(num_statements):
        answer.append(f"{days[i]}, {name} drives to {locations[location_choice[i]]} in a {cars[car_choice[i]]}.")
    
    # get days and cars for the answer
    days_per_car = {'SUV': [], 'minivan': []}
    for day, car, loc in zip(days, car_choice, location_choice):
        if loc == 0 and q_loc == 'grocery store':
            days_per_car[cars[car]].append(day)
        elif loc == 1 and q_loc == 'pharmacy':
            days_per_car[cars[car]].append(day)
    
    if q_car == 'minivan' and len(days_per_car['minivan']) > 0 and len(days_per_car['SUV']) == 0:
        answer.append(f"The answer is yes.")
    elif q_car == 'minivan' and len(days_per_car['SUV']) > 0:
        answer.append(f"The answer is no.")
    if q_car == 'SUV' and len(days_per_car['SUV']) > 0 and len(days_per_car['minivan']) == 0:
        answer.append(f"The answer is yes.")
    elif q_car == 'SUV' and len(days_per_car['minivan']) > 0:
        answer.append(f"The answer is no.")
    answer = ' '.join(answer)
    
    return unsegmented_document, segmented_document, question, answer


def task12_sample():
    """Generates a sample (unsegmented document, segmented documents, question, answer) for Task 12 "After how many x does y happen?"

    Task: Ask after how many days does someone join the person.
    
    Document example:
        [Task 12] Tom's Company
        Monday, Tom is alone.
        Tuesday, Tom is alone.
        Wednesday, Alice arrives.
        Thursday, Tom is with Alice.
        Friday, Tom is with Alice.

    Q/A example:
        [Task 12] After how many days does Alice join Tom?
        Monday, Tom is alone. Tuesday, Tom is alone. Wednesday, Alice arrives. Thursday, Tom is with Alice. Friday, Tom is with Alice. The answer is 2.
        
    Returns:
        str, List[str], str, str: The unsegmented document, segmented documents, question and answer.
    """
    name1 = get_unique_name()
    name2 = get_unique_name()
    
    num_statements = np.random.randint(3, 5)
    
    arrive = np.random.randint(1, num_statements+1)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday'][:num_statements]
    
    # unsegmented document
    unsegmented_document = [f"[Task 12] {name1}'s Company"]
    arrived = False
    for i, day in enumerate(days):
        if arrived:
            unsegmented_document.append(f"{day}, {name1} is with {name2}.")
        elif arrive == i:
            unsegmented_document.append(f"{day}, {name2} arrives.")
            arrived = True
        else:
            unsegmented_document.append(f"{day}, {name1} is alone.")
    unsegmented_document = '\n'.join(unsegmented_document)
    
    # segmented document
    segmented_document = []
    arrived = False
    for i, day in enumerate(days):
        if arrived:
            segmented_document.append(f"[Task 12] {name1}'s Company, Part {i+1}/{len(days)}\n{day}, {name1} is with {name2}.")
        elif arrive == i:
            segmented_document.append(f"[Task 12] {name1}'s Company, Part {i+1}/{len(days)}\n{day}, {name2} arrives.")
            arrived = True
        else:
            segmented_document.append(f"[Task 12] {name1}'s Company, Part {i+1}/{len(days)}\n{day}, {name1} is alone.")
    
    # question & answer
    question = f"[Task 12] After how many days does {name2} join {name1}?"
    
    answer = []
    arrived = False
    for i, day in enumerate(days):
        if arrived:
            answer.append(f"{day}, {name1} is with {name2}.")
        elif arrive == i:
            answer.append(f"{day}, {name2} arrives.")
            arrived = True
        else:
            answer.append(f"{day}, {name1} is alone.")
    if arrive >= num_statements:
        answer.append(f"The answer is never.")
    else:
        answer.append(f"The answer is {arrive+1}.")
        
    answer = ' '.join(answer)
    
    return unsegmented_document, segmented_document, question, answer


def task13_sample():
    """Generates a sample (unsegmented document, segmented documents, question, answer) for Task 13 "Is x the y'th in the list?"

    Task: Ask if a person is x'th on the list.
    
    Document example:
        [Task 13] Tom's Friends
        Tom meets Eve in the morning.
        Tom meets Alice at noon.
        Tom meets Bob in the afternoon.

    Q/A example:
        [Task 13] Is Bob the second person Tom meets?
        Tom meets Eve in the morning. Tom meets Alice at noon. Tom meets Bob in the afternoon. Bob is the third. The answer is no.
        
    Returns:
        str, List[str], str, str: The unsegmented document, segmented documents, question and answer.
    """
    person = get_unique_name()
    
    num_statements = np.random.randint(3, 5)
    friends = ['Person A', 'Person B', 'Person C', 'Person D', 'Person E']
    random.shuffle(friends)
    friends = friends[:num_statements]
    hours = ['in the morning', 'at noon', 'in the afternoon', 'in the evening'][:num_statements]
    
    # unsegmented document
    unsegmented_document = [f"[Task 13] {person}'s Friends"]
    for friend, hour in zip(friends, hours):
        unsegmented_document.append(f"{person} meets {friend} {hour}.")
    unsegmented_document = '\n'.join(unsegmented_document)
    
    # segmented document
    segmented_document = []
    for i, (friend, hour) in enumerate(zip(friends, hours)):
        segmented_document.append(f"[Task 13] {person}'s Friends, Part {i+1}/{len(friends)}\n{person} meets {friend} {hour}.")
    
    # question & answer
    order = ['first', 'second', 'third', 'fourth']
    random.shuffle(order)
    choice = order[0]
    random_friend = friends[np.random.randint(0, len(friends))]
    question = f"[Task 13] Is {random_friend} the {choice} person {person} meets?"
    
    answer = []
    for friend, hour in zip(friends, hours):
        answer.append(f"{person} meets {friend} {hour}.")

    friend_rank = 0
    for i, friend in enumerate(friends):
        if friend == random_friend:
            friend_rank = i
            break

    if friend_rank == 0:
        answer.append(f"{random_friend} is the first. The answer is {'yes' if choice == 'first' else 'no'}.")
    elif friend_rank == 1:
        answer.append(f"{random_friend} is the second. The answer is {'yes' if choice == 'second' else 'no'}.")
    elif friend_rank == 2:
        answer.append(f"{random_friend} is the third. The answer is {'yes' if choice == 'third' else 'no'}.")
    elif friend_rank == 3:
        answer.append(f"{random_friend} is the fourth. The answer is {'yes' if choice == 'fourth' else 'no'}.")
    answer = ' '.join(answer)
    
    return unsegmented_document, segmented_document, question, answer


def task14_sample():
    """Generates a sample (unsegmented document, segmented documents, question, answer) for Task 14 "Among the list of x, is there y?"

    Task: Ask if a person ate a certain fruit.
    
    Document example:
        [Task 14] Tom's Snacks
        Tom ate an apple at 8am.
        Tom ate a pear at 10am.
        Tom ate an orange at 2pm.

    Q/A example:
        [Task 14] Among the snacks that Tom ate, is there a banana?
        Tom ate an apple at 8am, a pear at 10am and an orange at 2pm. The answer is no.
        
    Returns:
        str, List[str], str, str: The unsegmented document, segmented documents, question and answer.
    """
    name = get_unique_name()
    
    num_statements = np.random.randint(2, 5)
    
    fruits = ['an apple', 'a pear', 'an orange', 'a banana', 'a cherry']
    random.shuffle(fruits)
    random_fruits = fruits[:num_statements]
    
    times = np.array(['8am', '10am', '12pm', '2pm'])
    random_times = np.zeros(times.shape[0], dtype=bool)
    random_times[:num_statements] = 1
    np.random.shuffle(random_times)
    random_times = times[random_times]
    
    # unsegmented document
    unsegmented_document = [f"[Task 14] {name}'s Snacks"]
    for fruit, t in zip(random_fruits, random_times):
        unsegmented_document.append(f"{name} ate {fruit} at {t}.")
    unsegmented_document = '\n'.join(unsegmented_document)
    
    # segmented document
    segmented_document = []
    for i, (fruit, t) in enumerate(zip(random_fruits, random_times)):
        segmented_document.append(f"[Task 14] {name}'s Snacks, Part {i+1}/{len(random_fruits)}\n{name} ate {fruit} at {t}.")
    
    # question & answer
    q_fruit = fruits[np.random.randint(0, len(fruits))]
    question = f"[Task 14] Among the snacks that {name} ate, is there {q_fruit}?"
    
    answer = []
    for fruit, t in zip(random_fruits, random_times):
        answer.append(f"{name} ate {fruit} at {t}.")
        
    if q_fruit in random_fruits:
        answer.append("The answer is yes.")
    else:
        answer.append("The answer is no.")
    answer = ' '.join(answer)
    
    return unsegmented_document, segmented_document, question, answer


def task15_sample():
    """Generates a sample (unsegmented document, segmented documents, question, answer) for Task 15 "Among the list of x, is there only y?"

    Task: Ask if only got As for certain courses.
    
    Document example:
        [Task 15] Tom's Grades
        Tom got an A in English.
        Tom got an A in Spanish.
        Tom got an B in Biology.
        Tom got an A in Physics.

    Q/A example:
        [Task 15] Did Tom only get A in science courses?
        Tom got an A in English. Tom got an A in Spanish. Tom got an B in Biology. Tom got an A in Physics. The answer is no.
        
    Returns:
        str, List[str], str, str: The unsegmented document, segmented documents, question and answer.
    """
    name = get_unique_name()
    
    lan_courses = ['English', 'Spanish', 'French']
    science_courses = ['Biology', 'Physics', 'Chemistry']
    lan_grades = ['an A' if np.random.rand() < 0.5 else 'a B' for _ in range(3)]
    science_grades = ['an A' if np.random.rand() < 0.5 else 'a B' for _ in range(3)]
    num_statements1 = np.random.randint(2, 4)
    num_statements2 = np.random.randint(2, 4)
    
    random_choice = np.zeros(3, dtype=bool)
    random_choice[:num_statements1] = 1
    np.random.shuffle(random_choice)
    lan_courses = np.asarray(lan_courses)[random_choice]
    
    random_choice = np.zeros(3, dtype=bool)
    random_choice[:num_statements2] = 1
    np.random.shuffle(random_choice)
    science_courses = np.asarray(science_courses)[random_choice]
    
    # unsegmented document
    unsegmented_document = [f"[Task 15] {name}'s Grades"]
    for i in range(num_statements1):
        unsegmented_document.append(f"{name} got {lan_grades[i]} in {lan_courses[i]}.")
    for i in range(num_statements2):
        unsegmented_document.append(f"{name} got {science_grades[i]} in {science_courses[i]}.")
    unsegmented_document = '\n'.join(unsegmented_document)
    
    # segmented document
    segmented_document = []
    for i in range(num_statements1):
        segmented_document.append(f"[Task 15] {name}'s Grades, Part {i+1}/{num_statements1 + num_statements2}\n{name} got {lan_grades[i]} in {lan_courses[i]}.")
    for i in range(num_statements2):
        segmented_document.append(f"[Task 15] {name}'s Grades, Part {i+1+num_statements1}/{num_statements1 + num_statements2}\n{name} got {science_grades[i]} in {science_courses[i]}.")
    
    # question & answer
    grade = 'A' if np.random.rand() < 0.5 else 'B'
    
    answer = []
    for i in range(num_statements1):
        answer.append(f"{name} got {lan_grades[i]} in {lan_courses[i]}.")
    for i in range(num_statements2):
        answer.append(f"{name} got {science_grades[i]} in {science_courses[i]}.")
    
    if np.random.rand() < 0.5: # science
        question = f"[Task 15] Did {name} only get {grade} in science courses?"
        if num_statements2 == 2:
            if science_grades[0][-1] == grade and science_grades[1][-1] == grade:
                answer.append(f"The answer is yes.")
            else:
                answer.append(f"The answer is no.")
        elif num_statements2 == 3:
            if science_grades[0][-1] == grade and science_grades[1][-1] == grade and science_grades[2][-1] == grade:
                answer.append(f"The answer is yes.")
            else:
                answer.append(f"The answer is no.")
    else: # language
        question = f"[Task 15] Did {name} only get {grade} in language courses?"
        if num_statements1 == 2:
            if lan_grades[0][-1] == grade and lan_grades[1][-1] == grade:
                answer.append(f"The answer is yes.")
            else:
                answer.append(f"The answer is no.")
        elif num_statements1 == 3:
            if lan_grades[0][-1] == grade and lan_grades[1][-1] == grade and lan_grades[2][-1] == grade:
                answer.append(f"The answer is yes.")
            else:
                answer.append(f"The answer is no.")
    answer = ' '.join(answer)
    
    return unsegmented_document, segmented_document, question, answer


def task16_sample():
    """Generates a sample (unsegmented document, segmented documents, question, answer) for Task 16 "Is x the same as y?"

    Task: Ask if went as many times to the beach and to the cinema.
    
    Document example:
        [Task 16] Tom's Activities
        Monday, Tom went to the beach.
        Tuesday, Tom went to the beach.
        Wednesday, Tom went to the cinema.
        Thursday, Tom went to the park.
        Friday, Tom went to the cinema.

    Q/A example:
        [Task 16] Did Tom go to the beach as many days as to the cinema?
        Monday, Tom went to the beach. Tuesday, Tom went to the beach. Wednesday, Tom went to the cinema. Thursday, Tom went to the park. Friday, Tom went to the cinema. The answer is yes.
        
    Returns:
        str, List[str], str, str: The unsegmented document, segmented documents, question and answer.
    """
    name = get_unique_name()
    
    num_statements = np.random.randint(4, 6)
    
    count1 = np.random.randint(1, num_statements // 2 + 1)
    count2 = np.random.randint(1, num_statements // 2 + 1)
    count3 = num_statements - count2 - count1
    activities = []
    for _ in range(count1):
        activities.append('beach')
    for _ in range(count2):
        activities.append('cinema')
    for _ in range(count3):
        activities.append('park')
    random.shuffle(activities)
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    random_days = np.zeros(len(days), dtype=bool)
    random_days[:num_statements] = 1
    np.random.shuffle(random_days)
    days = np.asarray(days)[random_days]
    
    # unsegmented & segmented document
    unsegmented_document = [f"[Task 16] {name}'s Activities"]
    segmented_document = []
    for i in range(num_statements):
        unsegmented_document.append(f"{days[i]}, {name} went to the {activities[i]}.")
        segmented_document.append(f"[Task 16] {name}'s Activities, Part {i+1}/{num_statements}\n{days[i]}, {name} went to the {activities[i]}.")
    unsegmented_document = '\n'.join(unsegmented_document)
    
    # question & answer
    question = f"[Task 16] Did {name} go to the beach as many days as to the cinema?"
    
    answer = []
    for i in range(num_statements):
        answer.append(f"{days[i]}, {name} went to the {activities[i]}.")
    
    beach_days = []
    cinema_days = []
    for activity, day in zip(activities, days):
        if activity == 'beach':
            beach_days.append(day)
        elif activity == 'cinema':
            cinema_days.append(day)
        
    if count2 == 0:
        answer.append(f"The answer is {'yes' if len(beach_days) == len(cinema_days) else 'no'}.")
    elif count2 == 1:
        answer.append(f"The answer is {'yes' if len(beach_days) == len(cinema_days) else 'no'}.")
    elif count2 == 2:
        answer.append(f"The answer is {'yes' if len(beach_days) == len(cinema_days) else 'no'}.")
    answer = ' '.join(answer)
    
    return unsegmented_document, segmented_document, question, answer


def task17_sample():
    """Generates a sample (unsegmented document, segmented documents, question, answer) for Task 17 "What is the state of x when y happens?"

    Task: Ask what clothes a person is wearing when the storm starts.
    
    Document example:
        [Task 17] Tom's Outfits
        8am, Tom is wearing a pyjama.
        10am, Tom is wearing workout clothes.
        12pm, Tom is wearing a bathrobe.
        2pm, the storm starts.
        4pm, Tom is wearing a raincoat.

    Q/A example:
        [Task 17] What was Tom wearing when the storm started?
        8am, Tom is wearing a pyjama. 10am, Tom is wearing workout clothes. 12pm, Tom is wearing a bathrobe. 2pm, the storm starts. 4pm, Tom is wearing a raincoat. The answer is a bathrobe.
        
    Returns:
        str, List[str], str, str: The unsegmented document, segmented documents, question and answer.
    """
    name = get_unique_name()
    
    num_statements = np.random.randint(4, 6)
    
    clothing = ['a pyjama', 'workout clothes', 'a bathrobe', 'a raincoat']
    random.shuffle(clothing)
    storm_start = np.random.randint(1, num_statements)
    
    times = ['8am', '9am', '10am', '11am', '12pm', '1pm', '2pm', '3pm', '4pm', '5pm']
    random_times = np.zeros(len(times), dtype=bool)
    random_times[:num_statements] = 1
    np.random.shuffle(random_times)
    random_times = np.asarray(times)[random_times]
    
    # unsegmented document & segmented document
    unsegmented_document = [f"[Task 17] {name}'s Outfits"]
    segmented_document = []
    clothing_count = 0
    for j, t in enumerate(random_times):
        if j == storm_start:
            unsegmented_document.append(f"{t}, the storm starts.")
            segmented_document.append(f"[Task 17] {name}'s Outfits, Part {j+1}/{len(random_times)}\n{t}, the storm starts.")
        else:
            unsegmented_document.append(f"{t}, {name} is wearing {clothing[clothing_count]}.")
            segmented_document.append(f"[Task 17] {name}'s Outfits, Part {j+1}/{len(random_times)}\n{t}, {name} is wearing {clothing[clothing_count]}.")
            clothing_count += 1
    
    # question & answer
    question = f"[Task 17] What was {name} wearing when the storm started?"
    answer = unsegmented_document[1:]
    answer.append(f"The answer is {clothing[storm_start - 1]}.")
    answer = ' '.join(answer)
    
    unsegmented_document = '\n'.join(unsegmented_document)
    
    return unsegmented_document, segmented_document, question, answer


def task18_sample():
    """Generates a sample (unsegmented document, segmented documents, question, answer) for Task 18 "If x had/hadn't happened, would y have happened?"

    Task: Ask if would have x dollars a certain day if had not sold a certain item.
    
    Document example:
        [Task 18] Tom's Money
        Monday, Tom sold a pencil for 2$.
        Tuesday, Tom sold an eraser for 1$.
        Wednesday, Tom sold a marker for 3$.
        Thursday, Tom sold a staple for 1$.

    Q/A example:
        [Task 18] If Tom hadn't sold a staple, would they have 6$ on Wednesday?
        Monday, Tom sold a pencil for 2$. Tuesday, Tom sold an eraser for 1$. Wednesday, Tom sold a marker for 3$. Thursday, Tom sold a staple for 1$. 2 + 1 + 3 = 6. The answer is yes.
        
    Returns:
        str, List[str], str, str: The unsegmented document, segmented documents, question and answer.
    """
    name = get_unique_name()
    
    num_statements = np.random.randint(3, 5)
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday']
    days_choice = days[:num_statements]
    
    items = ['a pencil', 'an eraser', 'a marker', 'a staple']
    random_choice = np.zeros(len(items), dtype=bool)
    random_choice[:num_statements] = 1
    np.random.shuffle(random_choice)
    items_choice = np.asarray(items)[random_choice]
    
    prices = [np.random.randint(1, 4) for _ in range(num_statements)]
    
    # unsegmented & segmented document
    unsegmented_document = [f"[Task 18] {name}'s Money"]
    segmented_document = []
    for i, (day, item, price) in enumerate(zip(days_choice, items_choice, prices)):
        unsegmented_document.append(f"{day}, {name} sold {item} for {price}$.")
        segmented_document.append(f"[Task 18] {name}'s Money, Part {i+1}/{len(prices)}\n{day}, {name} sold {item} for {price}$.")
    unsegmented_document = '\n'.join(unsegmented_document)
    
    # question & answer
    q_item = items_choice[np.random.randint(0, len(items_choice))]
    q_money = np.random.randint(3, 9)
    q_day = days_choice[np.random.randint(1, len(days_choice))]
    question = f"[Task 18] If {name} hadn't sold a {q_item}, would they have {q_money}$ on {q_day}?"
    
    answer = []
    for day, item, price in zip(days_choice, items_choice, prices):
        answer.append(f"{day}, {name} sold {item} for {price}$.")
    
    day_price_item = []
    for day, item, price in zip(days_choice, items_choice, prices):
        day_price_item.append((day, price, item))
        if day == q_day:
            break
        
    summation = []
    total = 0
    for day, price, item in day_price_item:
        if item != q_item:
            summation.append(str(price))
            total += price
    summation = ' + '.join(summation) + f" = {total}"
    answer.append(f"{summation}. The answer is {'yes' if total == q_money else 'no'}.")
    answer = ' '.join(answer)
    
    return unsegmented_document, segmented_document, question, answer


def get_unique_name():
    assert len(character_names) > 0
    return character_names.pop()


def main():
    args = parse_args()

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    with open('./names.txt', 'r') as file:
        character_names = [line.rstrip() for line in file]
    random.shuffle(character_names)

    unsegmented_documents = []
    segmented_documents = []
    qa_examples = [] # list of tuples: task #, question, answer
    eval_questions = [] # list of tuples: story id, task #, question, answer

    tasks_to_generate = [False] * 18
    for task_num in args.tasks:
        if 1 <= task_num <= 18:
            tasks_to_run[task_num - 1] = True

    for task, generate in enumerate(tasks_to_generate, 1):
        if generate:
            # Val and Test samples
            for _set in ['val', 'test']:
                for _ in range(eval(f"args.{_set}_samples_per_task")):
                    document_id = len(unsegmented_documents)
                    unsegmented_document, segmented_document, question, answer = eval(f"task{task}_sample")()
                    unsegmented_documents.append(unsegmented_document)
                    segmented_documents.append(segmented_document)
                    eval_questions.append((_set, f'{document_id:04d}', task, question, answer))
                
            # Q/A examples
            for _ in range(args.qa_samples_per_task):
                _, _, question, answer = eval(f"task{task}_sample")()
                qa_examples.append((task, question, answer))


    os.makedirs(args.dataset_dir)

    # unsegmented documents
    unsegmented_documents_dir = os.path.join(args.dataset_dir, 'unsegmented_documents/')
    os.makedirs(unsegmented_documents_dir)
    for i, unsegmented_doc in enumerate(unsegmented_documents):
        with open(os.path.join(unsegmented_documents_dir, f'{i:04d}.txt'), 'w') as f:
            f.write(unsegmented_doc)
            
    # segmented documents
    segmented_documents_dir = os.path.join(args.dataset_dir, 'segmented_documents/')
    os.makedirs(segmented_documents_dir)
    for i, segmented_doc in enumerate(segmented_documents):
        for j, part in enumerate(segmented_doc):
            with open(os.path.join(segmented_documents_dir, f'{i:04d}part{j:02d}.txt'), 'w') as f:
                f.write(part)
                
    # QA Examples
    pd.DataFrame(qa_examples).to_csv(os.path.join(args.dataset_dir, 'qa_examples.csv'), sep='|', index=False, header=['task', 'question', 'answer'])

    # Evaluation questions
    pd.DataFrame(eval_questions).to_csv(os.path.join(args.dataset_dir, 'eval_questions.csv'), sep='|', index=False, header=['set', 'story id', 'task', 'question', 'answer'])


if __name__ == '__main__':
    main()