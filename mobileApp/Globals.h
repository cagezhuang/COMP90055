//
//  Globals.h
//  AIR
//
//  Created by kage Zhuang on 25/2/19.
//  Copyright © 2019 Google. All rights reserved.
//

#ifndef Globals_h
#define Globals_h

#import <Foundation/Foundation.h>
#include <string>

extern NSString* model_file_name;
extern NSString* model_file_type;
extern NSString* labels_file_name;
extern NSString* labels_file_type;

extern int wanted_input_width;
extern int wanted_input_height;
extern int wanted_input_channels;
extern float input_mean;
extern float input_std;

extern std::string input_layer_name;
extern std::string output_layer_name;

#endif /* Globals_h */
