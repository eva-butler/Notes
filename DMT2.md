# DMT2 Notes



# Cardinality (Lecture 4 - class)
9/5/2024
- comparing the sizes of sets, specifically infinite sers
- Do all infinite sets have the same size - No
- Pigeonhole principle: If you have n holes, m pigeons, and m > n, then there must be atleast one hole that has more than one pigeon. Often used to compare the sizes of sets

Bijective functions:
- A proof by bijection
- Example
  <img width="397" alt="image" src="https://github.com/user-attachments/assets/ac021ad1-b502-4aa6-9185-6fcc790478c1">
  - if we can find a bijection between these two sets, then we can argue that the sets are the same size
  - the first formula is a conversion from a bit string to a natural number base 10
<img width="487" alt="image" src="https://github.com/user-attachments/assets/b4dddf33-e6e1-49fb-8727-874373153034">
  - first prove that the function is injective; argue that the differences in the input will naturally lead to different outputs
  - the sum of any lower order bit is strictly less than any higher order bit based on the powers of 2
  - On the right, we a proving that every single string in the output is mapped to an input
  - mistake: "convert into a bitstring as per the function on the previous slide"
  - Any natural number on the right side is mapped to an individual bit string
  - ON the right we are arguing that a bit greater than n would be larger than the largest natural number that you are trying to build as an output. Therefore, you know that every natural number in the output is mapped to a bitstring input
  - Example 2:
    <img width="424" alt="image" src="https://github.com/user-attachments/assets/7baa550b-70b9-4724-ba5a-1ce01e475f11">
    - the goal is to prove a bijection between the subsets and the bitstrings
    - 2^|S| <- |S| = n = 3 , {0,1}^n = 2^n
    - Using the bitstrings as a way to represent which items from the finite set, S, are present are not present.
    - Let's say S = {a,b,c} , 100 would map to a, ect. maps to which items are present
   <img width="504" alt="image" src="https://github.com/user-attachments/assets/4d54f567-285d-42d0-a4f4-45e07b61f9e1">
  - If two arbitrary subsets are different, then there is atleast one item in the set that differs in the two sets. Therefore we know that a bit would be different in the bitstring that is representing the presence of items in the set
  - As stated in the second purple part, there is a mapping to a subset of S that pairs to a String

Infinite Cardinality
- How do we compare the size of two infinite sets
    - we can say that two sets A and B have the same size fi there is a bijection between them
- Countability
  - A set S is countable if |S| <= |Natural Numbers|
  - If |S| = |Natural Numbers|, then S is "countably infinite"
  - A set is countable if there is an onto  (surjective) function from N to S
 Example:
  <img width="317" alt="image" src="https://github.com/user-attachments/assets/4ce1bea1-4e24-4718-be9b-1c62c3468444">
  - * means that you have all the 0 length strings, 1 length strings, ect
  -  is this countable? yes
  -  If we were to map to their natural numbers, we would have the problem of an inproper bijectiong because multiple inputs would map to a single output
  -  <img width="526" alt="image" src="https://github.com/user-attachments/assets/8bc4935d-a230-4ed4-9a76-908667c75f79">
- You can rather map the number box to a natural number. There is no bitstring that is missed. You are counting the boxes in a systamatic order and you will not skip anything. You will go on and on and on and map directly to the Natural numbers
- QUIZ -> a visual proof would be enough. COULD You turn it into a functino in a programming language.
- <img width="344" alt="image" src="https://github.com/user-attachments/assets/235092a9-c86c-4713-8733-9f4721438e42">
- ******COME back to this and make sure can write it out formally
- injective: different strings map to different numbers, different strings map to different nodes in the tree
- surjective: every number appears

Examples:
  <img width="476" alt="image" src="https://github.com/user-attachments/assets/e0097554-c059-4b20-8201-bda72e5eb4a3">
- bottom two are possible homework/quiz
- 1) Z+
  -  Z+ : 1,2,3,4, ....
  -  N: 0,1,2,3,4, ...
  -  if there exists a bijection between to the steps then they are the same size
  -  you can forever map 0->1, 1->2, ect, they are countably infinite
- 2) even
  - even numers: 0, 2,4,6,8 ..
  - natural: 0,1,2,3,4 ...
  - easy mapping: 0-0, 2-1, 4-2, 6-3, ect
  - given the natural you can multiply by 2
- 3) - odd numbers are the same thing then figure out the formula
- 4) Z
  - Z: ..., -2, -1, 0, 1, 2, ....
  - N: 0, 1,2,3,4 ...
  - mapping: alternate negative and positive
    - 0-0, 1-1, 2- -1, 3-2, 4- -3,
    - If you continue in this fashion, every number in Z will have a mapping to a natural number
 - 5) NxN
   - NxN: a bunch of tuples (x,y) is there a way to map the tuples to the naturals
   - <img width="369" alt="image" src="https://github.com/user-attachments/assets/44c3c7b3-829f-499e-8aed-c1c47a9b8f5a">
   - what is the systematic way to count these?
   - creating outter rings -> figure out what the formula is to represent this
   - also could do diagnals -> formula would probably be easiest

   REALLY JUST NEED TO FIGURE OUT A SYSTAMATIC WAY TO CREATE A BIJECTION AND CREATE A FUNCTION

- Q: QUIZZZZZZZ

**Number of Programs as Number of Functions**
- the set of programs you can write is countably infinite
- the number of functions you can write is uncountably infinite
- the set of functions is larger than the set of programs

- what is a python program? - a bunch of strings every python program is a member of sigma star.
- |sigma star| = |N| }we have shown this before, you can systamatically count these from the smallest ones to the largest ones

<img width="539" alt="image" src="https://github.com/user-attachments/assets/911715ff-ff29-4d55-8931-4791120233ca">
- how would you be able to map sigma star to sigma star.

How do you show something is uncountable?
- If you can show a subset is uncountable, then the superset must also be uncountable
- <img width="269" alt="image" src="https://github.com/user-attachments/assets/152aa110-15d3-4d71-b821-c65ad598c7f8">
- try to count the number of ways you can match the bit strings to true and false
- the inputs are the same everytime and we have to figure out what the outputs would be , they are changing
- The real thing we are counting is the number of possible bitstrings of infinite length that are binary
- If there is no onto mapping N to {0,1}^infinity, then uncountable
- best wat to do this is a proof by contradition. if the set is countable, then we can assume there must be a bijection. basically we need to prove that there is not a bijection
- We need to prove that there is a combination to prove is missing
- <img width="467" alt="image" src="https://github.com/user-attachments/assets/70799c9a-fc33-41a2-9d9e-3c975ddd3dba">
- take the bolded bits, and flip them, then we can make the claim, that the elements of the diagnal flipped do not appear in the mappings
- the element in the diagnol will ALWAYSSSS be different, you can prove that the items of the dignol and prove that it cannot exist. the element on the diagnol MUST not match. there is no mapping of infinite length bitstrings to the natural numbers. We can conclude the set of infinite length bit strings is uncountably infinite. 





