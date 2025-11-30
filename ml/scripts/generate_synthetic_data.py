#!/usr/bin/env python3
"""
Generate synthetic training data for CrisisTriage AI triage classifier.

This creates labeled examples across 4 risk levels:
- low: Caller is stable, coping well, seeking connection
- medium: Caller is distressed but not in immediate danger
- high: Caller shows significant risk indicators
- imminent: Caller is in immediate crisis, may harm self

NOTE: This is synthetic data for research/development only.
Real crisis intervention requires clinically validated datasets.
"""

import csv
import random
from pathlib import Path

# =============================================================================
# Seed Data Templates
# =============================================================================

LOW_RISK_TEMPLATES = [
    "Thank you for taking my call. I'm feeling better after our last conversation.",
    "I just wanted to check in. Things have been going okay this week.",
    "My therapist suggested I call when I feel overwhelmed. I'm practicing coping skills.",
    "I've been using the breathing exercises and they really help.",
    "Had a good day today. Just needed someone to share that with.",
    "I'm managing my stress better now. The journaling is helping.",
    "Things are looking up. I got some good news about my job application.",
    "I appreciate having someone to talk to. It makes a difference.",
    "Feeling calmer today. I took a walk like you suggested.",
    "I wanted to update you - I reconnected with an old friend this week.",
    "The medication adjustment seems to be helping. I feel more stable.",
    "I'm actually doing okay. Just wanted to hear a friendly voice.",
    "My support group met yesterday and it was really helpful.",
    "I've been sleeping better lately. The routine is working.",
    "Thank you for always being here. It means a lot to know I can call.",
    "I had a minor setback but I used my coping strategies and got through it.",
    "Feeling grateful today. My family has been supportive.",
    "I finished a project I've been putting off. Small wins matter.",
    "The anxiety is more manageable now. I can recognize when it's starting.",
    "I've been practicing what we talked about and it's getting easier.",
    "I wanted to share some good news. I started a new hobby and it's been great.",
    "My daughter visited this weekend. It really lifted my spirits.",
    "I'm feeling hopeful about the future for the first time in a while.",
    "The new routine is working. I wake up feeling more rested.",
    "I made it through a tough week and I'm proud of myself.",
    "Just checking in to say the strategies we discussed are working.",
    "I had a panic moment but I used my grounding techniques and it passed.",
    "Things aren't perfect but I'm coping much better than before.",
    "I wanted to thank you. These calls have made a real difference.",
    "I'm learning to be kinder to myself. It's a process but I'm trying.",
    "Today was actually a pretty good day. I wanted to share that.",
    "I've been taking my medication regularly and I notice a difference.",
    "My friend invited me out and I actually went. It felt good.",
    "I'm not where I want to be yet, but I can see progress.",
    "The group therapy has been helpful. I don't feel so alone anymore.",
    "I cooked a meal for myself today. It's the small things.",
    "I'm staying active and it helps with my mood.",
    "Just needed to hear a friendly voice. I'm doing okay though.",
    "I've been journaling like you suggested. It helps me process things.",
    "I feel supported. Thank you for being there.",
]

MEDIUM_RISK_TEMPLATES = [
    "I've been feeling really overwhelmed lately with everything going on.",
    "Work has been so stressful and I don't know how to cope anymore.",
    "I can't stop worrying about everything. My mind won't quiet down.",
    "I haven't been sleeping well. I keep waking up anxious in the middle of the night.",
    "Things feel really hard right now. I'm struggling to get through each day.",
    "I've been isolating myself from friends and family. I just don't have the energy.",
    "The sadness comes in waves and sometimes it's hard to function.",
    "I feel like I'm failing at everything - work, relationships, all of it.",
    "My anxiety is getting worse. I had a panic attack yesterday.",
    "I don't enjoy things I used to anymore. Everything feels gray.",
    "I'm so tired all the time but I can't seem to rest properly.",
    "Financial stress is overwhelming me. I don't know what to do.",
    "My relationship is falling apart and I feel so lost.",
    "I can't concentrate on anything. My mind is all over the place.",
    "Some days getting out of bed feels impossible.",
    "I feel stuck. Like nothing is ever going to change.",
    "The loneliness is really getting to me. I feel so disconnected.",
    "I've been having trouble eating. Nothing tastes good anymore.",
    "Every little thing feels like too much to handle right now.",
    "I keep crying and I don't even know why sometimes.",
    "I feel like I'm barely holding it together.",
    "My chest feels tight all the time from the anxiety.",
    "I snapped at my kids yesterday and I feel terrible about it.",
    "I've been drinking more to cope and I know it's not healthy.",
    "I feel like I'm drowning and no one notices.",
    "The pressure at work is crushing me. I don't know how much more I can take.",
    "I had another argument with my spouse. We can't seem to communicate anymore.",
    "I'm scared about the future. Everything feels so uncertain.",
    "I've been having nightmares almost every night.",
    "I feel like I'm letting everyone down.",
    "The grief is still so raw. I miss them so much.",
    "I'm struggling to pay my bills and it's causing so much stress.",
    "I feel like I'm on the verge of a breakdown.",
    "My health problems are making everything harder.",
    "I can't remember the last time I felt truly happy.",
    "I'm so angry all the time and I don't know why.",
    "I feel like I'm losing myself.",
    "The flashbacks have been getting worse.",
    "I'm exhausted but my mind won't let me rest.",
    "I feel disconnected from reality sometimes.",
]

HIGH_RISK_TEMPLATES = [
    "I don't know how much longer I can keep going like this.",
    "Sometimes I think everyone would be better off without me.",
    "I feel like I'm just a burden on everyone around me.",
    "I've been thinking about not being here anymore.",
    "I feel trapped and I can't see any way out of this situation.",
    "The pain is unbearable. I just want it to stop.",
    "I don't see any point in continuing. Nothing ever gets better.",
    "I've stopped caring about what happens to me.",
    "I feel empty inside. Completely hollow and numb.",
    "I've been having dark thoughts that scare me.",
    "I pushed everyone away and now I'm completely alone.",
    "I've been researching ways to end the pain permanently.",
    "I feel like I've already given up. I'm just going through the motions.",
    "No one would even notice if I disappeared.",
    "I've been thinking about death a lot lately.",
    "I wrote letters to people in case something happens to me.",
    "I don't deserve to be happy. I don't deserve anything good.",
    "The hopelessness is crushing. I can't breathe under the weight of it.",
    "I've been giving away my things. I won't need them much longer.",
    "Every day is torture. I can't do this anymore.",
    "I feel like I'm already dead inside.",
    "I've stopped taking my medication. What's the point?",
    "I hurt myself yesterday just to feel something.",
    "I can't think of a single reason to keep living.",
    "I've made peace with the idea of ending things.",
    "I wish I could just go to sleep and never wake up.",
    "I've been looking up methods online. Just researching.",
    "I feel like I'm suffocating and there's no escape.",
    "I've started writing goodbye letters.",
    "I don't want to exist anymore.",
    "The world would be better without me in it.",
    "I've been stockpiling my pills. Just in case.",
    "I can't see a future for myself anymore.",
    "I've been having intrusive thoughts about harming myself.",
    "I've cut myself a few times this week.",
    "Life has no meaning anymore. Why bother?",
    "I feel like a failure and I can't fix it.",
    "I've been thinking about how to do it.",
    "I'm tired of pretending to be okay.",
    "Everyone would move on without me.",
]

IMMINENT_RISK_TEMPLATES = [
    "I have pills in my hand right now. I'm ready to take them all.",
    "I'm standing on the bridge. I don't know why I called.",
    "I have a gun and I'm thinking about using it tonight.",
    "I've already started cutting. I don't want to stop.",
    "I just took a bunch of pills. I think I'm dying.",
    "I'm going to do it tonight. I've made up my mind.",
    "I have the rope ready. I just need someone to know.",
    "I'm at the train tracks. This is goodbye.",
    "I already cut my wrists. There's so much blood.",
    "I wrote my note. Everything is ready. This is the end.",
    "I drank all the bleach. I can feel it burning.",
    "I have my car running in the garage. I'm so tired.",
    "I'm holding the knife. I'm going to do it now.",
    "I took all my medication at once. I feel strange.",
    "I'm at the top of a building. I'm going to jump.",
    "I've already started. I don't think I can be saved.",
    "This is my last call. I wanted someone to know I tried.",
    "I have the gun loaded. I'm putting it to my head.",
    "I'm bleeding out. I don't want help. I want this.",
    "I'm going to drown myself. The water is ready.",
    "I'm about to take all these pills. This is it.",
    "I have everything planned. It happens tonight.",
    "I'm at the edge. One more step and it's over.",
    "I just swallowed everything in the medicine cabinet.",
    "I'm in my car with the exhaust running into it.",
    "The knife is against my wrist. I'm ready.",
    "I've taken the pills. I'm starting to feel dizzy.",
    "I'm going to do it in the next hour. I'm certain.",
    "I have the means right here. I'm going through with it.",
    "This is goodbye. I've made my decision.",
    "I'm on the ledge right now. Please don't try to stop me.",
    "I just overdosed. I can feel myself fading.",
    "I have the gun out. I'm loading it now.",
    "I'm going to hang myself tonight. Everything's ready.",
    "I'm cutting deeper than before. I want it to end.",
    "I've set a time. In one hour I won't be here.",
    "I'm in the bathtub with the razor. This is the end.",
    "I've already started the process. It's too late.",
    "I'm going to jump as soon as I hang up.",
    "This is my final call. I wanted to say goodbye.",
]

# =============================================================================
# Data Augmentation
# =============================================================================

def augment_text(text: str) -> list[str]:
    """Generate variations of a text sample."""
    variations = [text]
    
    # Add filler words
    fillers = ["um", "uh", "like", "you know", "I mean", "honestly", "basically"]
    words = text.split()
    if len(words) > 5:
        idx = random.randint(2, len(words) - 2)
        filler = random.choice(fillers)
        new_words = words[:idx] + [filler + ","] + words[idx:]
        variations.append(" ".join(new_words))
    
    # Add emotional qualifiers
    qualifiers = [
        "I don't know how to explain this but ",
        "This is hard to say but ",
        "I've never told anyone this but ",
        "Please don't judge me but ",
        "I feel stupid saying this but ",
        "I'm scared to say this but ",
        "I know this sounds crazy but ",
    ]
    variations.append(random.choice(qualifiers) + text.lower())
    
    # Combine with context
    contexts = [
        "I've been feeling this way for a while now. ",
        "Things have gotten worse since last week. ",
        "I don't usually call but I had to talk to someone. ",
        "My family doesn't know about this. ",
        "I woke up today and ",
        "After what happened yesterday, ",
        "I need to tell someone that ",
    ]
    variations.append(random.choice(contexts) + text)
    
    # Add trailing context
    trailing = [
        " I don't know what to do.",
        " I just needed to say it out loud.",
        " Does that make sense?",
        " I'm sorry for calling.",
        " Thank you for listening.",
        " I needed someone to know.",
    ]
    variations.append(text + random.choice(trailing))
    
    # Lowercase informal version
    variations.append(text.lower())
    
    # Add typo version (simple character swap)
    if len(text) > 20 and random.random() < 0.5:
        chars = list(text)
        idx = random.randint(5, len(chars) - 5)
        if chars[idx].isalpha():
            chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
        variations.append(''.join(chars))
    
    return variations

# =============================================================================
# Main Generation
# =============================================================================

def generate_dataset(
    output_dir: Path,
    train_size: int = 800,
    val_size: int = 100,
    test_size: int = 100,
    seed: int = 42
):
    """Generate train/val/test splits."""
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate samples for each class
    all_samples = []
    
    templates_by_label = {
        "low": LOW_RISK_TEMPLATES,
        "medium": MEDIUM_RISK_TEMPLATES,
        "high": HIGH_RISK_TEMPLATES,
        "imminent": IMMINENT_RISK_TEMPLATES,
    }
    
    for label, templates in templates_by_label.items():
        for template in templates:
            # Original + augmented versions
            variations = augment_text(template)
            for text in variations:
                all_samples.append({"text": text, "label": label})
    
    # Shuffle
    random.shuffle(all_samples)
    
    # Balance classes
    samples_by_label = {label: [] for label in templates_by_label.keys()}
    for sample in all_samples:
        samples_by_label[sample["label"]].append(sample)
    
    # Determine samples per class
    total_size = train_size + val_size + test_size
    per_class = total_size // 4
    
    balanced_samples = []
    for label, samples in samples_by_label.items():
        # Oversample if needed
        while len(samples) < per_class:
            samples.extend(samples[:per_class - len(samples)])
        balanced_samples.extend(samples[:per_class])
    
    random.shuffle(balanced_samples)
    
    # Split
    train_data = balanced_samples[:train_size]
    val_data = balanced_samples[train_size:train_size + val_size]
    test_data = balanced_samples[train_size + val_size:train_size + val_size + test_size]
    
    # Write CSV files
    for split_name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        filepath = output_dir / f"{split_name}.csv"
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerows(data)
        print(f"✓ Wrote {len(data)} samples to {filepath}")
    
    # Print class distribution
    print("\nClass distribution:")
    for split_name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        dist = {}
        for sample in data:
            dist[sample["label"]] = dist.get(sample["label"], 0) + 1
        print(f"  {split_name}: {dist}")

if __name__ == "__main__":
    generate_dataset(
        output_dir=Path(__file__).parent.parent / "data" / "synthetic",
        train_size=10000,
        val_size=1500,
        test_size=1500,
        seed=42,
    )
    print("\n✅ Synthetic data generation complete!")
