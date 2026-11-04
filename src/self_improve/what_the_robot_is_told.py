"""The words the robot is given: the home, the job, its three tools, and why notes help.

Every prompt in the arm that reads its own record is built from the blocks here, so the
whole of what the model is told sits in one place a person can read straight through.

Five rules for anything added here, all of them from things that went wrong:

  - short sentences and ordinary words. No invented phrases. If a sentence needs three
    commas, it needs to be two sentences.
  - FEW EXAMPLES, AND SAY THEY ARE EXAMPLES. This model takes an example as an
    instruction: given a list of the kinds of thing to notice, it wrote the things on the
    list. Each example we give is a hypothesis we have put in its head instead of letting
    it find one, so the list is short and it is told plainly that the real range is wider.
  - NOTHING THAT DESCRIBES THIS STUDY'S DISRUPTION. An earlier draft offered "they have
    been at home all week when they are normally out" as an example of something worth
    noticing. That is the thing we are testing whether it can notice. It is cut.
  - nothing the robot could not have seen for itself. The people's names are the one thing
    that looks like an exception and is not: the objects are named after their owners, so
    "mug_yuki" is in every prompt already. Which resident is which person is NOT said,
    because the robot's own record says "resident_1" and never says more.
  - the instruction first and the reason after it. Measured: with the reason first this
    model answered a night of six fresh sightings with an empty list of edits, every time.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

# Spelled out rather than imported: memory_notes imports nothing from here and this
# file imports nothing from it, and the pair must stay that way.
memory_notes_A_PROFILE_AND_TOLD = ("the log and notes, a profile of each person, told on the first night")


def the_people_who_live_here(asked_objects: Sequence[str],
                             resident_ids: Sequence[str]) -> List[str]:
    """Who lives here, and that they are real people. Two examples, named as examples.

    `resident_ids` is used only to count the people. It used to be printed, with a sentence
    asking the robot to work out which identifier was which person - a puzzle about our data
    and not about the household, and one that cost it attention for nothing. The looks now
    name people directly: see who_lives_here.
    """
    names = sorted({o.rsplit("_", 1)[1].capitalize() for o in asked_objects
                    if "_" in o and o.rsplit("_", 1)[1] != "shared"})
    who = (" and ".join(names) if len(names) < 3
           else ", ".join(names[:-1]) + " and " + names[-1])
    return [
        "This is a real home. "
        + {1: "One person lives", 2: "Two people live",
           3: "Three people live"}.get(len(resident_ids),
                                       f"{len(resident_ids)} people live")
        + " in it"
        + (f", and from the names on their things they are called {who}." if names
           else "."),
        "",
        "They are real people with real lives. They have habits and preferences. They "
        "make mistakes. They leave a thing wherever they happened to be standing. Their "
        "habits change when something in their life changes.",
        "",
        "So working out where a thing is really means working out what the person who "
        "owns it was doing. A thing is where somebody left it.",
        "",
        "There are many kinds of thing worth working out about people, and far more than "
        "anything written here could list. Two examples only, so you know the sort of "
        "thing: somebody might always take a cup with them when they move rooms, or "
        "somebody might be in a different room on a Saturday. Do not treat those two as "
        "the list. Look at what you actually saw and work out what is going on in this "
        "home.",
        "",
        "You can see the people when you walk into a room, and your record names them.",
    ]


def what_the_job_is(budget: int) -> List[str]:
    return [
        "YOUR JOB. Somebody asks you where one thing is, right now. You may walk into up "
        f"to {budget} rooms to look for it, one at a time. Then you name exactly one spot "
        "where you think it is.",
        "",
        "If you find the thing, say where you found it. If you run out of rooms, you still "
        "have to give your best answer.",
    ]


def the_three_things_you_have(budget: int) -> List[str]:
    return [
        "THE THREE THINGS YOU HAVE.",
        "",
        "1. Your eyes. When you walk into a room you see everything in it at the current "
        "moment. That also tells you what is not in that room, which is worth knowing too.",
        "",
        "2. Your record. Everything you have ever seen is written down for you and you "
        "never lose any of it. When you are asked about a thing, you are shown every time "
        "you have seen that thing, with the day and the time, where it was, and what else "
        "was in the room with it. You are also shown the rooms you have looked in without "
        "finding it.",
        "",
        "3. Your notes. These are the only thing you write yourself, and you choose what "
        "goes in them. You can add a note, change a note, or archive one. Nothing you "
        "write is ever thrown away.",
        "",
        "Your record tells you where things have been. It cannot tell you where a thing "
        "will be next, and that is what you are always asked. Your notes are the only "
        "place you can keep what you have worked out. It may help to use them for your "
        "own ideas about what is going on here, and for anything in what you saw that was "
        "surprising or important.",
    ]


def how_much_you_may_write(reason_characters: int) -> str:
    """The model was never told it gets cut off. Saying the number costs one line."""
    return (f"Keep your reasoning under {reason_characters} characters. Anything longer "
            f"is cut off, and a cut-off answer cannot be read.")


# The lines that set the role. This arm has its own, because the shared ones say "nobody
# tells the robot where anything is", which is a negation about an expectation a freshly
# started model does not have, and they are used by other arms whose prompts must not move.
CHOOSING_SYSTEM = ("You are a robot in somebody's home. You work out where things are by "
                   "looking, and you keep your own notes. Answer with JSON only.")
ANSWERING_SYSTEM = ("You are a robot in somebody's home. Use what you have seen and what "
                    "you have written down to say where a thing is. Answer with JSON "
                    "only.")
WRITING_SYSTEM = ("You are a robot in somebody's home. At the end of each day you write "
                  "down what you have worked out about the people who live there. Answer "
                  "with JSON only.")


# ---------------------------------------------------------------------------------
# EVERY WORD THAT DIFFERS BETWEEN THE FOUR WAYS OF KEEPING MEMORY IS IN THIS ONE DICT.
# ---------------------------------------------------------------------------------
#
# This exists because the two original ways of keeping memory were never given the same
# instructions, and nobody noticed until the run was half done. Seven differences, written
# up in results/self_improve/three_prompts/THE_WRITING_PROMPTS_WERE_NOT_THE_SAME.md, of
# which the worst were: the two arms were told OPPOSITE things about loss, one was told not
# to repeat itself and the other was not, and only one was asked to cite its sightings.
# Every difference we measured between the two styles was therefore confounded with a
# difference in what we told them.
#
# So the shape here is: everything shared is built by `the_nightly_prompt` below, and the
# ONLY thing a method may change is its entry in this dictionary. To add a way of keeping
# memory, add a key. If you find yourself wanting to change anything outside this dict for
# one method only, that is the confound coming back - stop and say so instead.
HOW_YOUR_MEMORY_WORKS: Dict[str, List[str]] = {
    # The profile-keeping arm. Its notes work exactly as our arm's do - the paragraph below is
    # word for word the same - and it has one more thing it keeps, described in `the_profiles`.
    "the log and notes, with a profile of each person": [
        "YOUR MEMORY has two parts.",
        "",
        "THE FIRST is a short PROFILE of each person who lives here. You were told their names "
        "and what they do, and nothing else; the rest is for you to work out and keep up to "
        "date. A profile is the place for what is true of a person over days and weeks: what "
        "they do with their time, what they seem to like, where they spend it, and WHAT IS "
        "GOING ON WITH THEM AT THE MOMENT if something is - a person can be away, or busy, or "
        "unwell, and that can last a while and then stop. Say when something started, and say "
        "when it has stopped. A profile is not a list of where things are.",
        "",
        "THE SECOND is a list of separate notes, each with its own number, which carries over "
        "from yesterday exactly as it was. Tonight you change it by writing edits: you can add "
        "a note, revise a note by its number, or attach today's sightings to a note as evidence "
        "for it or against it.",
        "",
        "Everything you have ever seen is also kept for you in a record, and you are shown all "
        "of it about anything you are asked. Repeating your record in a note adds nothing, so "
        "do not just copy it.",
        "",
        "WHY THERE ARE TWO PARTS. A note about where a thing lives gets narrowed every time you "
        "see that thing somewhere, which is right for a note and wrong for a person. If Tomas is "
        "unwell for a fortnight, that belongs in his profile, where it can stay true for a "
        "fortnight and then be written out - not in a note about a water bottle, where tomorrow's "
        "sighting will quietly delete it.",
        "",
        "Write the edits tonight calls for. Revising a note keeps what it said before, so "
        "nothing you write is ever lost.",
    ],

    "wholesale rewrite": [
        "YOUR MEMORY is one piece of writing about this home. The version you wrote last "
        "night is shown above. Tonight you write the whole thing again, and what you write "
        "replaces it as the only thing you will have tomorrow.",
        "",
        "You can keep as much of last night's version as you want, change any of it, or "
        "start again. Anything you leave out is gone.",
    ],

    "incremental edits": [
        "YOUR MEMORY is a list of separate notes, each with its own number. It carries "
        "over from yesterday exactly as it was. Tonight you change it by writing edits: "
        "you can add a note, revise a note by its number, or attach today's sightings to "
        "a note as evidence for it or against it.",
        "",
        "Write the edits tonight calls for. Revising a note keeps what it said before, so "
        "nothing you write is ever lost.",
    ],

    "claim store told if it was right": [
        "YOUR MEMORY is a list of separate notes, each with its own number, carried over "
        "from yesterday exactly as it was. Tonight you change it by writing edits: you "
        "can add a note, revise a note by its number, attach today's sightings to a note "
        "as evidence for it or against it, or join two notes into one.",
        "",
        "You have also been shown how your notes did today: which of them helped you find "
        "something and which of them sent you to the wrong room. It may help to act on "
        "the ones that misled you first.",
        "",
        "Write the edits tonight calls for. Revising a note keeps what it said before, so "
        "nothing you write is ever lost.",
    ],

    # This arm's block is now its control's block plus ONE paragraph: the record. Every
    # other difference between the two has been moved into the shared part, so a difference
    # in results can only be the record and the sentence about not copying it.
    "the log and notes about the routine": [
        "YOUR MEMORY is a list of separate notes, each with its own number. It carries "
        "over from yesterday exactly as it was. Tonight you change it by writing edits: "
        "you can add a note, revise a note by its number, or attach today's sightings to "
        "a note as evidence for it or against it.",
        "",
        "Everything you have ever seen is also kept for you in a record, and you are shown "
        "all of it about anything you are asked. Repeating your record in a note adds "
        "nothing, so do not just copy it.",
        "",
        "Write the edits tonight calls for. Revising a note keeps what it said before, so "
        "nothing you write is ever lost.",
    ],

    "a small working memory and an archive": [
        "YOUR MEMORY has two parts. Your WORKING MEMORY is small and has a fixed size, and "
        "everything in it is put in front of you every time you are asked anything. Your "
        "ARCHIVE has no size limit, and nothing in it is shown to you unless you search "
        "for it. You decide what goes where.",
        "",
        "Tonight you can write a new note, revise a note by its number, attach today's "
        "sightings to a note, move a note into your working memory, or move one out to the "
        "archive. Nothing is ever deleted: a note in the archive can always be found again "
        "by searching.",
        "",
        "Your working memory is full when it is full. If you try to put something in and "
        "there is no room, it will be refused and you will be told so. It may help to move "
        "something out to the archive first. What you keep in your working memory is what "
        "you will be thinking with tomorrow, so it is worth spending on the things you "
        "will need every day rather than on something you can look up.",
    ],

    "MemGPT as published": [
        "YOUR MEMORY has two parts. Your BLOCK is one piece of writing with a fixed size, and "
        "all of it is in front of you every time you are asked anything. Your ARCHIVE holds "
        "separate passages, and none of them is shown to you unless you search for it.",
        "",
        "Tonight you can do four things, as many times as you need. You can add a line to the "
        "end of your block. You can replace a piece of your block: name the piece exactly as "
        "it appears and say what goes in its place, and if you name something that is not "
        "there word for word, nothing happens and you are told so. You can put a passage in "
        "your archive. And you can search your archive, which answers a page at a time.",
        "",
        "To move something out of your block, put it in your archive first and then replace "
        "it in the block with nothing. Nothing is ever lost that way: a passage in the "
        "archive can always be found again by searching for it.",
        "",
        "Your block is full when it is full. A line that will not fit is refused and you are "
        "told how much room there is. What you keep in the block is what you will be thinking "
        "with tomorrow, so it is worth spending on what you need every day rather than on "
        "something you could look up.",
    ],
}


# Told and asked to commit: OUR ARM'S WORDS, UNCHANGED. The paragraph that asks it to commit is not
# here, because anything here appears on all thirty-two nights and it only has something to do on the
# two nights a sentence arrives. Put here, the arm would differ from its parent EVERY night and a
# difference between them could not be attributed to the prediction. It is passed in per night by
# `what_to_write_before_you_have_seen_it` instead, so on the other thirty nights the prompt is
# identical to the parent's - which also means those nights replay from the response cache for free.
HOW_YOUR_MEMORY_WORKS["the log and notes, told and asked what will change"] = (
    HOW_YOUR_MEMORY_WORKS["the log and notes about the routine"])


def what_to_write_before_you_have_seen_it() -> List[str]:
    """The paragraph for a night a sentence arrives and nothing has been seen of it yet."""
    return [
        "IF YOU HAVE BEEN TOLD SOMETHING ABOVE THAT YOU HAVE NOT SEEN FOR YOURSELF, that is the "
        "one thing tonight that you cannot check and must not wait to confirm. Before anything "
        "else, write what you expect to be DIFFERENT tomorrow because of it: which person will be "
        "somewhere other than usual, at which hours, and which of their things will therefore be "
        "in a different room, and which room. Name the rooms. You have seen none of this, so you "
        "will be guessing - guess anyway, because tomorrow you will be asked where things are "
        "before you have had a chance to look, and a guess written down tonight is the only thing "
        "you will have. Write each expectation as its own note, and say in its condition that it "
        "holds while the thing you were told is true, so that you can set it aside when it stops.",
        "",
    ]


# THE SAME WORDS FOR THE ARM THAT IS ALSO TOLD. Set here rather than written into the dict above so
# that the two arms cannot drift: they are the same list object, and the ONLY difference between the
# arms is the one sentence delivered on one night. A copy is how arms drift apart, and this project
# has paid for that twice.
HOW_YOUR_MEMORY_WORKS[memory_notes_A_PROFILE_AND_TOLD] = HOW_YOUR_MEMORY_WORKS[
    "the log and notes, with a profile of each person"]


# What one piece of this method's memory is called, so the sentence about being cut off
# reads correctly. The wholesale arm writes one long piece; the others write short notes.
WHAT_ONE_PIECE_IS_CALLED: Dict[str, str] = {
    "wholesale rewrite": "Your writing is cut off after {n} characters.",
    "incremental edits": "Each note is cut off after {n} characters, so keep one idea to "
                         "a note.",
    "claim store told if it was right":
        "Each note is cut off after {n} characters, so keep one idea to a note.",
    "the log and notes about the routine":
        "Each note is cut off after {n} characters, so keep one idea to a note.",
    "a small working memory and an archive":
        "Each note is cut off after {n} characters, so keep one idea to a note.",
    "MemGPT as published":
        "Anything you write is cut off after {n} characters.",
}


def how_many_notes_you_may_write(n: int) -> List[str]:
    """The budget, said out loud, because until 2026-09-26 it never was.

    The schema has always stopped the model at its allowance, and the prompt has never named
    the number - so a night that wanted more was TRUNCATED rather than rationed. Our own arm sat
    exactly at its cap of sixteen on 27% of nights at 24 questions a day and on a third of the
    disrupted nights, and never knew the cap was there.

    That makes a cap sweep measure the wrong thing. "Does a smaller memory trade steady accuracy
    for faster adaptation" is a question about RATIONING - about a writer choosing what is worth
    a slot - and blind truncation is not rationing, it is loss. So the number is now said, twice:
    what the budget is, and that it binds, and that choosing what to spend it on is part of the
    job.
    """
    return [f"TONIGHT YOU MAY WRITE OR CHANGE AT MOST {n} NOTES. That is a hard limit: the "
            f"{n + 1}th is thrown away, so anything you leave out is lost and anything you write "
            f"that was not worth a slot has cost you one.",
            "",
            "So decide what is worth the space before you write. If more has happened than fits, "
            "keep what will still be true tomorrow and what you would be most wrong about "
            "without it."]


def the_nightly_prompt(method: str, household, day: int, the_day_in_words: str,
                       memory_as_it_stands: str, characters_a_note: int,
                       name_the_things_it_is_asked_about: bool = False,
                       a_message_tonight: Optional[str] = None,
                       how_many_notes: Optional[int] = None,
                       extra_before_the_instruction: Sequence[str] = ()) -> List[str]:
    """The whole of what any arm is told at the end of a day.

    Everything here is the same for every way of keeping memory. The one thing that
    differs is `HOW_YOUR_MEMORY_WORKS[method]`. That is the point of the function: it makes
    the seven accidental differences that confounded the first comparison impossible to
    reintroduce without deleting a line of this file.
    """
    if method not in HOW_YOUR_MEMORY_WORKS:
        raise ValueError(f"no words written for {method!r}; "
                         f"known: {sorted(HOW_YOUR_MEMORY_WORKS)}")
    return (
        the_people_who_live_here(household.asked_objects, household.resident_ids)
        + (["", *how_many_notes_you_may_write(how_many_notes)] if how_many_notes else [])
        + ["", f"It is the end of day {day}."]
        + (["", a_message_tonight] if a_message_tonight else [])
        + ["",
           "The rooms of this home and the spots in each:",
           *[f"- {room}: {', '.join(household.places_in_room[room])}"
             for room in household.rooms],
           "",
           ("The things you are asked about: " + ", ".join(household.asked_objects)
            if name_the_things_it_is_asked_about else
            "Somebody will ask you where a thing in this home is. Nobody has said which "
            "things, so it could be any of them."),
           "",
           "WHAT YOU SAW TODAY. Each look lists everything that was in the room at that "
           "moment, so anything not listed was not there:",
           "",
           the_day_in_words,
           "",
           "YOUR MEMORY AS IT STANDS:",
           "",
           memory_as_it_stands,
           ""]
        + list(extra_before_the_instruction)
        + HOW_YOUR_MEMORY_WORKS[method]
        + ["",
           # THE ADVICE IS SHARED, and it was not. Four of these six sentences used to sit
           # in the record-reading arm's own block, so that arm got advice on how to think
           # and its own control got one imperative - which means a win for it could have
           # been the six sentences rather than the design. Found by the wave agent
           # comparing the blocks before launching. Advice about how to think belongs to
           # every arm; only how the memory is stored belongs to a method.
           "It may help to think about what was surprising or important in what you saw "
           "today, and why it happened. What you write down is your own thinking, not a "
           "copy of what you saw.",
           "",
           "It may help to say when something you write is true, if it is only true at "
           "some times. That is what lets you choose between two things you have written "
           "that disagree. Write the condition itself, not the date you happened to see "
           "it: what has to be the case for it to be true.",
           "",
           "It may help to write down what you would expect to see if something you "
           "believe turned out to be wrong.",
           "",
           "When something you believed stops being true, make your best judgement on "
           "whether to keep it, change it, or set it aside. Setting it aside is its own "
           "thing to do, separate from changing what it says: it means the note is not "
           "true at the moment, and it stays in your memory so you can bring it back. "
           "Changing the words of a note leaves it standing. It may help to record what "
           "made you decide.",
           "",
           # Shared, and it is shared deliberately: asking only one arm to cite its
           # sightings was one of the seven differences.
           "You can point at any sighting using the number in square brackets beside it.",
           "",
           WHAT_ONE_PIECE_IS_CALLED[method].format(n=characters_a_note)]
    )


def the_profiles(profiles: Dict[str, str], who_lives_here: str) -> List[str]:
    """What the profile-keeping arm is shown about the people, and what it must do with it.

    On the first night `profiles` is empty and only `who_lives_here` is known - the names and what
    each person does, which is what somebody would tell a robot on the day it arrived. No age: the
    households do not carry one, so there is none to give. Nothing about hobbies, chores, bedrooms
    or workspaces is handed over, because those are what the robot is supposed to work out.
    """
    lines = ["WHO LIVES HERE, as you were told on your first day:", "", who_lives_here, ""]
    if profiles:
        lines += ["YOUR PROFILE OF EACH PERSON, as it stands:", ""]
        lines += [f"- {name}: {text}" for name, text in sorted(profiles.items())]
    else:
        lines += ["YOUR PROFILE OF EACH PERSON: you have not written one yet."]
    lines += ["",
              "First, write each person's profile as it should stand tonight. Give the whole "
              "profile for each person, not a change to it: what you write replaces what is "
              "there. Keep it short. If nothing about a person has changed, write theirs again "
              "as it was.",
              ""]
    return lines
