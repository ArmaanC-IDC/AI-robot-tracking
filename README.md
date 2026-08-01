# FRC AI Scouting & Multi-Object Tracking System

This project is an automated video tracking system for the FIRST Robotics Competition (FRC), developed as part of an ongoing research project. It takes match video feeds, detects the robots on the field, and records their continuous movement paths. 

The primary challenge in FRC video tracking is that standard algorithms frequently lose or swap robot identities when robots collide, drive behind field obstacles, or share identical alliance bumper colors. To solve this without manual correction, this project uses a two-step tracking pipeline that combines basic physics predictions with an AI visual similarity model.

---

## Table of Contents
- [Project Overview](#project-overview)
- [How the Tracker Works](#how-the-tracker-works)
- [System Architecture](#system-architecture)
- [Data Collection & Annotation Tools](#data-collection--annotation-tools)
- [The Re-Identification (Re-ID) Model](#the-re-identification-re-id-model)
- [Project Structure](#project-structure)

---

## Project Overview

When scouting FRC matches from video, recording accurate movement paths requires keeping continuous track of up to six robots moving at high speeds. Because robots on the same alliance look similar from a distance and frequently bump into each other, basic bounding-box trackers fail. 

This software solves identity swapping by layering three different verification methods: checking where the robot should physically be, calculating collision probabilities when paths cross, and using a machine learning model to visually match images of a robot before and after it gets lost. This methodology and its accuracy metrics are being documented for a formal research paper.

---

## How the Tracker Works

When the video stream detects a robot, the software assigns it to a continuous path using a three-step decision process:

1. **Step 1 (Speed & Direction Check):** The system calculates where a robot could possibly move based on maximum acceleration and velocity. If a detected robot appears inside that expected zone and no other robots are around, the path updates instantly.
2. **Step 2 (Visual Re-Identification):** If robots collide for an extended period or get completely hidden behind field structures, spatial math cannot guarantee identity. The software crops the image of the mystery robot and passes it to a trained AI model (a Siamese Neural Network). This model compares the new image against past images of the known robots and identifies it based on visual features, rather than just physical position.

---

## Data Collection & Annotation Tools

Training a custom AI model to recognize specific FRC robots requires thousands of image samples. To automate this without manual file sorting, the project includes custom dataset tools:

- **Interactive OpenCV Dashboard:** Pauses the match video and displays a grid of auto-cropped robot images detected within the field boundaries. A user can accept or discard training images with a single click.
- **Dataset Manager:** Scripts that automatically organize harvested robot images into training folders while preserving original files, preventing accidental data loss during large-scale exports.

---

## The Re-Identification (Re-ID) Model

The visual identification model is built using Keras (with MobileNetV2) and PyTorch (using OSNet). 

- **Why a Siamese Network:** Instead of a standard classifier that just memorizes a fixed list of robots, this model is trained to produce embeddings between images. It calculates the visual similarity between any two images using the distance between their embeddings. This allows the system to accurately match robots it has never seen before by comparing them to a temporary gallery built during the match.
- **Data Augmentation:** The training pipeline applies controlled brightness adjustments, rotations, and contrast changes to prevent overfitting.

---
