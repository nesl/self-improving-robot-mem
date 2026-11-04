# Day-1 prompts: `claim store plus the log` against `incremental edits`, hh_s2_t03, 8 a day

## search (room choice)
schema identical: True; max_tokens identical: True

```diff
--- incremental edits
+++ claim store plus the log
@@ -30,4 +30,10 @@
 - storage: never looked
 
+YOUR RECORD. You do not have to remember where things are: everything you have ever seen is kept for you, and here is all of it for the thing you were asked about.
+
+You have never seen towel_tomas.
+
+Rooms you have never looked in at all: balcony, bathroom, bedroom_1, dining, entry, kitchen, living, office, storage
+
 Your notes:
 
```

## answer
schema identical: True; max_tokens identical: True

```diff
--- incremental edits
+++ claim store plus the log
@@ -18,4 +18,16 @@
 from looks it took.
 
+YOUR RECORD. You do not have to remember where things are: everything you have ever seen is kept for you, and here is all of it for the thing you were asked about.
+
+You have never seen water_bottle_tomas.
+
+Rooms you have looked in without finding it:
+- dining: 1 look(s), most recently day 1 at 09:22
+- living: 1 look(s), most recently day 1 at 09:22
+- entry: 1 look(s), most recently day 1 at 09:22
+- bathroom: 1 look(s), most recently day 1 at 07:17
+
+Rooms you have never looked in at all: balcony, bedroom_1, kitchen, office, storage
+
 These are its notes:
 
```

## nightly update (night 1)
schema identical: True; max_tokens identical: True

```diff
```
