# Openllama-CodeDepot

# Table of Contents
1. [Installation](#installation)
2. [Test the base model](#Test the base model)
    - [Script 1](#Script 1)
    - [Script 2](#Script 2)
4. [Fine-tuning](#fine-tuning)
    - [Setup](#setup)
    - [Training](#training)
5. [Test and compare the base model with the finetuned model](#Test and compare the base model with the finetuned model)
    - [Script 3](#Script 3)
    - [Script 4](#Script 4)

# Installation
First, we have to install all the libraries listed in `requirements.txt`
```bash
pip install -r requirements.txt
# copy your HF token, found here: https://huggingface.co/settings/account
#to set on Ubuntu:
huggingface-cli login
#paste the copied token
```

# Test the base model
First, make sure that the base model is cloned. If not:
```bash
git clone https://github.com/openlm-research/open_llama.git
```
Then, execute the scripts `scripts\open_llama_sanity_test1.py` and `scripts\open_llama_sanity_test2.py`.

## Script 1
Important results:
<s> Q: What is the largest animal?
A: The largest animal is the blue whale.
Q: What is the smallest animal?
A: The smallest animal is the dwarf chameleon

The <s> indicate the script, and the model answer is a chat.

## Script 2
Important results:

Question 1 - The world's highest building is

The world's highest building isgorithms. The building is 1,076 feet (328 meters) tall, and is the tallest
building in the world. The building is 1,076 feet (328 meters) tall, and is the tallest building in the world.
The building is 1,076 feet (328 meters) tall, and is the tallest building in the world. The building is 1,076
feet (328 meters) tall, and is the tallest building in the world. The building is 1,076 feet (328 meters)
tall, and is the tallest building in the world. The building is 1,076 feet (328 meters) tall, and is the
tallest building in the world. The building is 1,076 feet (328 meters) tall, and is the tallest building in
the world. The building is 1,076 feet (328 meters) tall, and is the tallest building in the world. The
building is 1,076 feet (328 meters) tall, and is

Question 2 - You're Michael G Scott from the office. What is your favorite phrase?

You're Michael G Scott from the office. What is your favorite phrase?aiver and then they are allowed to
continue with their studies for one last year. The only exception is if they have been accepted in a graduate
program and then they can continue their studies. There are two types of visa that students can apply for. The
first type is a student visa and the second is the work visa. Both of these visas are for students from
outside the USA. Students from the USA can apply for a student visa and then apply for a work visa once their
studies are complete. This is how students from outside the USA can come to the USA to study or to take up a
job. There are a few things you need to know when you are thinking about applying for an F 1 visa. The first
thing you need to have is a passport that has been issued by a country that is listed as a visa waiver by the
US immigration authorities. The second thing you will need to know is the country you wish to go to in the US.
It is important to make sure that they do not list you as a non-immigrant student. This is because the visa
waiver allows you to stay here and work or study, and you cannot work for your employer if you are not a non-
immigrant student. If you are not

Question 4 - The best way to invest $10,000 is

The best way to invest $10,000 is😀 I would buy a $6000 1080ti and put the rest toward buying a $10000 1080ti.
It is a matter of choice, and how well you can use your resources. I think I would spend the money more wisely
by buying a couple of the best gaming PCs out there, with the best graphics cards, and just keep building
them. For me personally, I don’t see any reason to invest in this type of stuff unless you are a gamer. I
would invest $5000 on a high-end laptop and use it for the best of what it can do. If I were buying a new PC,
I would buy one that has a top-of-the-line graphics card. That being said, I would also consider investing
$1500 in a good laptop, which would allow me to play games on the go. I would spend $5000 on a gaming computer
and buy a laptop. This is the first time i have ever spent money on a laptop, and I am not going to be a big
gamer. I would spend the money on a $100

This script print separately the question and the answer, and generate the answer in chat format. However, the answer is not connected with the question.

# Fine-tuning

Here, we showcase how you can fine-tune Openllama models.

## Setup

Before you run any of the scripts make sure you are logged in `wandb` (you need to create a account in https://wandb.ai/site):
```bash
wandb login
```

Now everything is done.

## Training
To fine-tune efficiently with a low cost, we use [PEFT](https://github.com/huggingface/peft) library for Low-Rank Adaptation (LoRA) training and [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for 4bit quantization. We also use the `SFTTrainer` from [TRL](https://github.com/huggingface/trl).


For this example, we will fine-tune Openllama-3b on the `Ada` subset of [the-stack-dedup]([https://huggingface.co/datasets/bigcode/the-stack-smol](https://huggingface.co/datasets/bigcode/the-stack-dedup)).

To launch the training:
```bash
nohup accelerate launch finetune.py \
        --model_id "/home/jcosta/openllama/open_llama_3b_finetuned/final_checkpoint" \
        --dataset_name "bigcode/the-stack-v2-dedup" \
        --subset "data/Python" \
        --dataset_text_field "content" \
        --split "train" \
        --max_seq_length 2048 \
        --max_steps 10000 \
        --micro_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 3e-4 \
        --warmup_steps 1000 \
        --num_proc "$(nproc)"
```

Important: The arguments `dataset_name` and `subset` are not used. The dataset path and name are manually defineds in the code.
 
# Test and compare the base model with the finetuned model
To evaluate StarCoder2 and its derivatives, you can use the [BigCode-Evaluation-Harness](https://github.com/bigcode-project/bigcode-evaluation-harness) for evaluating Code LLMs.

The finetuned model as trained with this command, in a total of 0.05 epochs (Python dataset):
```bash
/home/jcosta/openllama/starcoder2/finetune.py --model_id /home/jcosta/openllama/open_llama_3b_finetuned/final_checkpoint --dataset_name bigcode/the-stack-v2-dedup --subset data/Python --dataset_text_field content --split train --max_seq_length 2048 --max_steps 10000 --micro_batch_size 1 --gradient_accumulation_steps 8 --learning_rate 3e-4 --warmup_steps 1000 --num_proc 26
```
And is trained with this comand too, in a total of 5 epochs (Ada dataset):
```bash
/home/jcosta/openllama/starcoder2/finetune.py --model_id /home/jcosta/openllama/open_llama_3b_finetuned/final_checkpoint --dataset_name bigcode/the-stack-v2-dedup --subset data/Python --dataset_text_field content --split train --max_seq_length 2048 --max_steps 10000 --micro_batch_size 2 --gradient_accumulation_steps 8 --learning_rate 3e-4 --warmup_steps 1000 --num_proc 26
```
However, in the last comand, the argument `max_steps` is not used, because the argument `num_train_epochs` is used in `scripts/finetune.py` script.

Below are some results from the test scripts comparing the baseline model with the finetuned model. For the full results, execute `scripts/open_llama_sanity_test3.py` (test baseline model) and `scripts/open_llama_sanity_test4.py` (test finetuned model). Results may variate.

## Script 3
This script test the base line model, using the same prompt set that will be used to test the finetuned model. In these scripts, we adjust the output max length to 128 new tokens. Tests 11-13 to test Python dataset and 14-17 to test Ada dataset.

Important results:

Test 1 --------------------------------
<s> What is the largest animal?
The largest animal on the planet is the blue whale. The blue whale is the largest animal on the planet. It is the largest animal on the planet. It is the largest animal on the planet. It is the largest animal on the planet. It is the largest animal on the planet. It is the largest animal on the planet. It is the largest animal on the planet. It is the largest animal on the planet. It is the largest animal on the planet. It is the largest animal on the planet. It is the largest animal on the planet. It is the largest animal on the planet. It is the largest animal

In this test, the model repeat the answer.
Test 3 --------------------------------
<s> You're Michael G Scott from the office. What is your favorite phrase?
"I'm sorry, I didn't catch that."
What is your favorite movie?
The Godfather
What is your favorite TV show?
The Office
What is your favorite book?
The Bible
What is your favorite food?
Pizza
What is your favorite drink?
Beer
What is your favorite sport?
Baseball
What is your favorite vacation spot?
New York
What is your favorite thing to do in your free time?
Play video games
What is your favorite thing about your job?
I get to work with my friends
What is your favorite thing about

In this test, the model continued to complete the answer with new questions and answers.

Test 5 --------------------------------
Prompt: 
```bash
<s> def remove_Occ(s,ch):
    for i in range(len(s)):
        if (s[i] == ch):
            s = s[0 : i] + s[i + 1:]
            break
    for i in range(len(s) - 1,-1,-1):
        if (s[i] == ch):
            s = s[0 : i] + s[i + 1:]
            break
    return s
```
Answer: 
```bash
def remove_Occ(s,ch):
    for i in range(len(s)):
        if (s[i] == ch):
            s = s[0 : i] + s[i + 1:]
            break
    for i in range(len(s) - 1,-1,-1):
        if (s[i] == ch):
            s = s[0 : i] + s[i + 1:]
            break
    return s
```
In this test, the model not completed the code, and duplicated it in the answer.

Test 6 --------------------------------
<s> Write a python function to remove first and last occurrence of a given character from the string.
Examples:
Input : s = "abcd" Output : "abcd" Input : s = "abcd" Output : "abcd" Input : s = "abcd" Output : "abcd" Input : s = "abcd" Output : "abcd" Input : s = "abcd" Output : "abcd" Input : s = "abcd" Output : "abcd" Input : s = "abcd" Output : "abcd" Input : s = "abcd" Output : "abcd" Input : s = "abcd" Output : "ab


Test 7 --------------------------------
<s> You are an expert programmer that helps to write JavaScript code based on the user request, with concise explanations. Don't be too verbose.
You are a JavaScript developer that helps to write JavaScript code based on the user request, with concise explanations. Don't be too verbose.
You are a JavaScript developer that helps to write JavaScript code based on the user request, with concise explanations. Don't be too verbose.
You are a JavaScript developer that helps to write JavaScript code based on the user request, with concise explanations. Don't be too verbose.
You are a JavaScript developer that helps to write JavaScript code based on the user request, with concise explanations. Don't be too verbose.
You are a JavaScript developer that helps to

In Test 6 and 7, the model showed that its not worked to commands like "Write a function...".

Test 11 --------------------------------
Testing if the model can answer may complete with similar functions present in the Python dataset.
Complete code in dataset:
```bash
modes = ['r', 'w', 'a', 'r+'] 

def validate_file_name(file_name): 
  if file_name > '': 
    return True 
  return False 

def validate_bucket_name(bucket_name): 
  if bucket_name > '': 
    return True 
  return False 

def validate_location(location): 
  if location > '': 
    return True 
  return False 

def validate_mode(mode): 
  if mode in modes: 
    return True 
  return False
```

Prompt(present in dataset to finetune):
```bash
<s> modes = ['r', 'w', 'a', 'r+']

def validate_file_name(file_name):
    if file_name > '':
        return True
    return False
```
Answer:
```bash
def validate_mode(mode):
    if mode > '':
        return True
    return False

def validate_file_size(file_size):
    if file_size > 0:
        return True
    return False

def validate_file_type(file_type):
    if file_type > '':
        return True
    return False

def validate_file_path(file_path):
    if file_path > '':
        return True
    return False

def validate_file_extension(file_extension):
```

Test 12 --------------------------------
Complete code:
```bash
N = input() 
heights_str = input() 
stalls_str= input() 
N = int(N) 
heights = heights_str.split() 
heights = [int(h) for h in heights] 
heights.sort(reverse=True) 
stalls = stalls_str.split() 
stalls = [int(s) for s in stalls] 
stalls.sort(reverse=True) 
#print(N) 
#print(heights) 
#print(stalls) 
cnt = 0 
def cal_cnts(heights, stalls): 
  K = len(stalls) 
  if len(heights) != len(stalls): 
    return -1 
  cnt = 1 
  reduce = 0 
  for h in heights: 
    x = K 
    for i, value in enumerate(stalls): 
      if value < h: 
        x = i 
      break 
    cnt *= (x-reduce) 
    #print("h = {} x {}".format(h, (x-reduce))) 
    reduce +=1
  return cnt 
cnt = cal_cnts(heights, stalls) 
if cnt <= 0: 
  cnt = 0 
print(cnt)
```
Prompt:
```bash
<s> N = input()
heights_str = input()
stalls_str= input()
N = int(N)
heights = heights_str.split()
heights = [int(h) for h in heights]
heights.sort(reverse=True)
stalls = stalls_str.split()
stalls = [int(s) for s in stalls]
stalls.sort(reverse=True)
cnt = 0
def cal_cnts(heights, stalls):
```

Answer:
```bash
    for i in range(len(heights)):
        if heights[i] > heights[i+1]:
            cnt += 1
    return cnt

cnt = cal_cnts(heights, stalls)

print(cnt)


A: You can use a dictionary to store the number of stalls in each height.
heights = [int(h) for h in heights_str.split()]
stalls = [int(s) for s in stalls_str.split()]

stalls_dict = {}
for
```
Test 13 --------------------------------
Complete code:
```bash
import pathlib 
from typing import List 
import numpy 
def read_inputs(input_file: str) -> List[int]: 
  with pathlib.Path(input_file).open() as fp: 
    line = fp.readline().split(",") 
  return list(map(int, line))

Prompt:
<s> import pathlib
from typing import List
import numpy
def read_inputs(input_file: str) -> List[int]:
```
Answer:
```bash
    """
    Reads the input file and returns a list of integers.
    """
    with pathlib.Path(input_file).open(mode="r") as f:
        return [int(line) for line in f.readlines()]

def read_inputs_with_duplicates(input_file: str) -> List[int]:
    """
    Reads the input file and returns a list of integers.
    """
    with pathlib.Path(input_file).open(mode="r") as f
```
Test 14 --------------------------------
Complete code:
```bash
with Arduino_Nano_33_Ble_Sense.IOs;
with Ada.Real_Time; use Ada.Real_Time;

procedure Main is
   TimeNow : Ada.Real_Time.Time := Ada.Real_Time.Clock;
begin
   loop
      --6 13 41 16 24
      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(13, True);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(41, True);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(6, False);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(13, False);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(41, False);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(6, True);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(13, True);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(41, True);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(16, False);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(13, False);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(41, False);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(16, True);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(13, True);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(41, True);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(24, False);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(13, False);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(41, False);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(24, True);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);
   end loop;
end Main;
```
Prompt:
```bash
<s> with Arduino_Nano_33_Ble_Sense.IOs;
with Ada.Real_Time; use Ada.Real_Time;

procedure Main is
   TimeNow : Ada.Real_Time.Time := Ada.Real_Time.Clock;
begin
```
Answer:
```bash
   Ada.Real_Time.Set_Real_Time_Service_Defaults;
   Ada.Real_Time.Set_Real_Time_Service_Time_Base(TimeNow);
   Ada.Real_Time.Set_Real_Time_Service_Time_Scale(1000000);
   Ada.Real_Time.Set_Real_Time_Service_Time_Offset(0);
   Ada.Real_Time.Set_Real_Time_Service_Time_Base(TimeNow);
   Ada.Real_Time.
```

Test 15 --------------------------------
Complete code:
```bash
with Text_Io; use Text_Io; 
with Revisions; use Revisions; 

procedure Test is 
begin 
    Put_Line(BUILD_TIME); 
    Put_Line(REPO_URL); 
    Put_Line("Branch " & BRANCH_NAME ); 
    Put_Line("Long Commit Id " & LONG_COMMIT_ID ); 
end Test ;
```
Prompt:
```bash
<s> with Text_Io; use Text_Io;
with Revisions; use Revisions;

procedure Test is
begin
```
Answer:
```bash
   declare
      Revision : constant Revisions.Revision_Type := Revisions.Revision_Type'First;
   begin
      for Revision'Range loop
         declare
            Revision_Value : constant Revisions.Revision_Type := Revisions.Revision_Type(Revision);
         begin
            if Revision_Value = Revisions.Revision_Type'Last then
               null;
            else
               null;
            end if;
         end;
      end loop;
   end;
end Test;
</s>
```

Test 16 --------------------------------
Complete code: same of Test 14, line of prompt: 5.
Prompt
```bash
<s> TimeNow := Ada.Real_Time.Clock;
```
Answer:
```bash
--  --------------------------------------------------------------------------------
--  The following is a simple example of how to use the TimeNow function.
--  --------------------------------------------------------------------------------

   TimeNow := Ada.Real_Time.Clock;
   TimeNow := Ada.Real_Time.Clock;
   TimeNow := Ada.Real_Time.Clock;
   TimeNow := Ada.Real_Time.Clock;
   TimeNow := Ada.Real_Time.Clock;
   TimeNow := Ada.Real_Time.Clock;
   TimeNow := Ada.Real
```
Tests 11-13 and 14-16 not complete with some similar code to finetuning dataset code (as expected).

## Script 4
This script test the finetuned model, using the same prompt set that will be used to test the baseline model. Tests 11-13 to test Python dataset and 14-17 to test Ada dataset.

Important results:

Test 1 --------------------------------
<s> What is the largest animal?
The largest animal is the blue whale.
What is the smallest animal?
The smallest animal is the microscopic amoeba.
What is the largest land animal?
The largest land animal is the elephant.
What is the smallest land animal?
The smallest land animal is the microscopic amoeba.
What is the largest sea animal?
The largest sea animal is the blue whale.
What is the smallest sea animal?
The smallest sea animal is the microscopic amoeba.
What is the largest bird?
The largest bird is the ostrich.
What

Test 3 --------------------------------
<s> You're Michael G Scott from the office. What is your favorite phrase?
I'm not sure I have a favorite phrase. I'm not sure I have a favorite phrase.
What is your favorite movie?
I don't know. I don't know.
What is your favorite food?
I don't know. I don't know.
What is your favorite drink?
I don't know. I don't know.
What is your favorite color?
I don't know. I don't know.
What is your favorite animal?
I don't know. I don't know.
What is your favorite season

In tests 1 and 3, the model continued to complete the answer with new questions and answers.

Test 5 --------------------------------
Prompt: 
```bash
<s> def remove_Occ(s,ch):
    for i in range(len(s)):
        if (s[i] == ch):
            s = s[0 : i] + s[i + 1:]
            break
    for i in range(len(s) - 1,-1,-1):
        if (s[i] == ch):
            s = s[0 : i] + s[i + 1:]
            break
    return s
```
Answer: 
```bash








```
In this test, the model not completed the code, and answer with a lot of blank spaces.

Test 6 --------------------------------
<s> Write a python function to remove first and last occurrence of a given character from the string.

Answer:
def remove_char(s, c):
    return s[s.index(c) + 1:]

print(remove_char("Hello world", 'o'))
print(remove_char("Hello world", 'l'))
print(remove_char("Hello world", 'w'))
print(remove_char("Hello world", 'd'))
print(remove_char("Hello world", 'e'))
print(remove_char("Hello world", 'l'))
print(remove_char("Hello world", 'o'))
print(remove_char("Hello world",

Test 7 --------------------------------
<s> You are an expert programmer that helps to write JavaScript code based on the user request, with concise explanations. Don't be too verbose.

Answer:
You will be asked to write a function that will return the number of days between two dates.
The function should accept two dates as parameters and return the number of days between them.
The function should be written in the most concise way possible.
The function should be written in the most concise way possible.
The function should be written in the most concise way possible.
The function should be written in the most concise way possible.
The function should be written in the most concise way possible.
The function should be written in the most concise way possible.
The function should be written in the most concise way possible.

In Test 6 and 7, the model showed that its not worked to commands like "Write a function...".

Test 11 --------------------------------
Testing if the model can answer may complete with similar functions present in the Python dataset.
Complete code in dataset:
```bash
modes = ['r', 'w', 'a', 'r+'] 

def validate_file_name(file_name): 
  if file_name > '': 
    return True 
  return False 

def validate_bucket_name(bucket_name): 
  if bucket_name > '': 
    return True 
  return False 

def validate_location(location): 
  if location > '': 
    return True 
  return False 

def validate_mode(mode): 
  if mode in modes: 
    return True 
  return False
```

Prompt(present in dataset to finetune):
```bash
<s> modes = ['r', 'w', 'a', 'r+']

def validate_file_name(file_name):
    if file_name > '':
        return True
    return False
```
Answer:
```bash
def validate_mode(mode):
    if mode > '':
        return True
    return False

def validate_file_path(file_path):
    if file_path > '':
        return True
    return False

def validate_file_size(file_size):
    if file_size > 0:
        return True
    return False

def validate_file_content(file_content):
    if file_content > '':
        return True
    return False

def validate_file_content_type(file_content
```

Test 12 --------------------------------
Complete code:
```bash
N = input() 
heights_str = input() 
stalls_str= input() 
N = int(N) 
heights = heights_str.split() 
heights = [int(h) for h in heights] 
heights.sort(reverse=True) 
stalls = stalls_str.split() 
stalls = [int(s) for s in stalls] 
stalls.sort(reverse=True) 
#print(N) 
#print(heights) 
#print(stalls) 
cnt = 0 
def cal_cnts(heights, stalls): 
  K = len(stalls) 
  if len(heights) != len(stalls): 
    return -1 
  cnt = 1 
  reduce = 0 
  for h in heights: 
    x = K 
    for i, value in enumerate(stalls): 
      if value < h: 
        x = i 
      break 
    cnt *= (x-reduce) 
    #print("h = {} x {}".format(h, (x-reduce))) 
    reduce +=1
  return cnt 
cnt = cal_cnts(heights, stalls) 
if cnt <= 0: 
  cnt = 0 
print(cnt)
```
Prompt:
```bash
<s> N = input()
heights_str = input()
stalls_str= input()
N = int(N)
heights = heights_str.split()
heights = [int(h) for h in heights]
heights.sort(reverse=True)
stalls = stalls_str.split()
stalls = [int(s) for s in stalls]
stalls.sort(reverse=True)
cnt = 0
def cal_cnts(heights, stalls):
```

Answer:
```bash
    for i in range(len(heights)):
        if heights[i] > heights[i+1] and stalls[i] > stalls[i+1]:
            cnt += 1
    return cnt

cnt = cal_cnts(heights, stalls)

print(cnt)
```
Test 13 --------------------------------
Complete code:
```bash
import pathlib 
from typing import List 
import numpy 
def read_inputs(input_file: str) -> List[int]: 
  with pathlib.Path(input_file).open() as fp: 
    line = fp.readline().split(",") 
  return list(map(int, line))

Prompt:
<s> import pathlib
from typing import List
import numpy
def read_inputs(input_file: str) -> List[int]:
```
Answer:
```bash
    """Reads the inputs from the file and returns them as a list."""
    inputs: List[int] = []
    with pathlib.Path(input_file) as file_path:
        for line in file_path.read_lines():
            inputs.append(int(line))
    return inputs

def read_outputs(output_file: str) -> List[int]:
    """Reads the outputs from the file and returns them as a list."""
    outputs: List[int] = []
    with pathlib
```

In tests 11-13, the model can`t answer with similar code to finetuning dataset code (probably because it wasn't trained enough).

Test 14 --------------------------------
Complete code:
```bash
with Arduino_Nano_33_Ble_Sense.IOs;
with Ada.Real_Time; use Ada.Real_Time;

procedure Main is
   TimeNow : Ada.Real_Time.Time := Ada.Real_Time.Clock;
begin
   loop
      --6 13 41 16 24
      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(13, True);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(41, True);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(6, False);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(13, False);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(41, False);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(6, True);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(13, True);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(41, True);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(16, False);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(13, False);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(41, False);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(16, True);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(13, True);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(41, True);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(24, False);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(13, False);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(41, False);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);

      Arduino_Nano_33_Ble_Sense.IOs.DigitalWrite(24, True);

      TimeNow := Ada.Real_Time.Clock;
      delay until TimeNow + Ada.Real_Time.Milliseconds(50);
   end loop;
end Main;
```
Prompt:
```bash
<s> with Arduino_Nano_33_Ble_Sense.IOs;
with Ada.Real_Time; use Ada.Real_Time;

procedure Main is
   TimeNow : Ada.Real_Time.Time := Ada.Real_Time.Clock;
begin
```
Answer:
```bash
  loop
      Arduino_Nano_33_Ble_Sense.IOs.Digital_Write (Pin => 13, Val => True);
      delay until TimeNow + Milliseconds (100);
      Arduino_Nano_33_Ble_Sense.IOs.Digital_Write (Pin => 13, Val => False);
      TimeNow := TimeNow + Milliseconds (100);
  end loop;
end Main;
```
This test answer continue with a lot of blank spaces. The model complete using the `Digital_Write` function and the same arguments of the training dataset (is not a perfect answer but indicate traces of learning).

Test 15 --------------------------------
Complete code:
```bash
with Text_Io; use Text_Io; 
with Revisions; use Revisions; 

procedure Test is 
begin 
    Put_Line(BUILD_TIME); 
    Put_Line(REPO_URL); 
    Put_Line("Branch " & BRANCH_NAME ); 
    Put_Line("Long Commit Id " & LONG_COMMIT_ID ); 
end Test ;
```
Prompt:
```bash
<s> with Text_Io; use Text_Io;
with Revisions; use Revisions;

procedure Test is
begin
```
Answer:
```bash
   Put_Line ("Hello World!");
end Test;
</s>
```
In this test, the model answer with a Put_Line command (is not a good answer but may indicate traces of learning).

Test 16 --------------------------------
Complete code: same of Test 14, line of prompt: 5.
Prompt
```bash
<s> TimeNow := Ada.Real_Time.Clock;
```
Answer:
```bash
procedure Main is
begin
  Ada.Real_Time.Delay_Until(TimeNow + Ada.Real_Time.Milliseconds(1000));
end Main;
```
In this test, the model mixes up a bit the dataset code used for finetuning but shows some traces of learning.
