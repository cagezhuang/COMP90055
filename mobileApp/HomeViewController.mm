//
//  HomeViewController.m
//  tflite_camera_example
//
//  Created by macbook on 12/2/19.
//  Copyright © 2019 Google. All rights reserved.
//

#import "HomeViewController.h"

@interface HomeViewController ()

@end

@implementation HomeViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    // initialize model to be inception v3
    self.modelLabel.text = model_file_name;
}

- (IBAction)modelSelectButtonClick:(id)sender {
    
    // Create actionsheet
    UIAlertController *actionSheet = [UIAlertController alertControllerWithTitle:@"Model Selection" message:@"Please select a model" preferredStyle:UIAlertControllerStyleActionSheet];
    
    // Inception model button clicked
    [actionSheet addAction:[UIAlertAction actionWithTitle:@"Inception v3" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action) {
        
        // Setup Inception v3 model variable
        model_file_name = @"Inception_v3";
        model_file_type = @"tflite";
        labels_file_name = @"Inception_v3_labels";
        labels_file_type = @"txt";
        
        // Input dimentsion and values
        wanted_input_width = 299;
        wanted_input_height = 299;
        wanted_input_channels = 3;
        input_mean = 128.0;
        input_std = 128.0;
        
        // Input and output tensor names
        input_layer_name = "Mul";
        output_layer_name = "final_result";
        
        self.modelLabel.text = model_file_name;
    }]];
    // Mobilenet Button clicked
    [actionSheet addAction:[UIAlertAction actionWithTitle:@"Mobilenet v1" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action) {
        
        // Setup Mobilenet v1 1.0 224 model variables
        model_file_name = @"Mobilenet_v1";
        model_file_type = @"tflite";
        labels_file_name = @"Mobilenet_v1_labels";
        labels_file_type = @"txt";
        
        // Input dimension and value
        wanted_input_width = 224;
        wanted_input_height = 224;
        wanted_input_channels = 3;
        input_mean = 127.5f;
        input_std = 127.5f;
        
        // Input and output tensor names
        input_layer_name = "input";
        output_layer_name = "final_result";
        
        self.modelLabel.text = model_file_name;
    }]];
    // Cancel button clicked
    [actionSheet addAction:[UIAlertAction actionWithTitle:@"Cancel" style:UIAlertActionStyleCancel handler:nil]];
    
    [self presentViewController:actionSheet animated:YES completion:nil];
}

@end
