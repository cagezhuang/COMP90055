//
//  Globals.m
//  AIR
//
//  Created by kage Zhuang on 25/2/19.
//  Copyright © 2019 Google. All rights reserved.
//
//  Description: This file is create to store global variables
//               such as model location, label location, etc
//

#import "Globals.h"
#import <Foundation/Foundation.h>

// the model is initially selected as Inception v3
NSString* model_file_name = @"Inception_v3";
// model file name
NSString* model_file_type = @"tflite";
// model lable file name
NSString* labels_file_name = @"Inception_v3_labels";
// label file type
NSString* labels_file_type = @"txt";

// These dimensions need to match those the model was trained with.
// Original setup for inception v3 model
int wanted_input_width = 299;
int wanted_input_height = 299;
int wanted_input_channels = 3;
float input_mean = 128.0;
float input_std = 128.0;

// Define the input and out tensor name. Original for incepition v3 model
std::string input_layer_name = "Mul";
std::string output_layer_name = "final_result";
