# Day-1 prompts: `claim store plus today's sightings` against `incremental edits`, hh_s2_t03, 8 a day

## search (room choice)
schema identical: True; max_tokens identical: True

```diff
--- incremental edits
+++ claim store plus today's sightings
@@ -30,4 +30,8 @@
 - storage: never looked
 
+TODAY'S RECORD. What you have seen earlier today of the thing you were asked about.
+
+You have not seen towel_tomas earlier today.
+
 Your notes:
 
```

## answer
schema identical: True; max_tokens identical: True

```diff
--- incremental edits
+++ claim store plus today's sightings
@@ -18,4 +18,11 @@
 from looks it took.
 
+TODAY'S RECORD. What you have seen earlier today of the thing you were asked about.
+
+You have not seen water_bottle_tomas earlier today.
+
+Rooms you have looked in earlier today without finding it:
+- bathroom: 1 look(s), most recently day 1 at 07:17
+
 These are its notes:
 
```

## nightly update (night 1)
schema identical: True; max_tokens identical: True

```diff
```
