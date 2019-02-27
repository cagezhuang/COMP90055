//
//  ImageViewController.m
//  AIR
//
//  Created by macbook on 12/2/19.
//  Copyright © 2019 Google. All rights reserved.
//

#import "ImageViewController.h"
#import <AssertMacros.h>
#import <AssetsLibrary/AssetsLibrary.h>
#import <CoreImage/CoreImage.h>
#import <ImageIO/ImageIO.h>

#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <queue>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/op_resolver.h"
#include "tensorflow/contrib/lite/string_util.h"

#define LOG(x) std::cerr

// The following code is original created by tensorflow iOS app example.
namespace {

    // Find the path of a file
    NSString* FilePathForResourceName(NSString* name, NSString* extension) {
        NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
        if (file_path == NULL) {
            LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "." << [extension UTF8String]
            << "' in bundle.";
        }
        return file_path;
    }

    // Load model labels for label file
    void LoadLabels(NSString* file_name, NSString* file_type, std::vector<std::string>* label_strings) {
        NSString* labels_path = FilePathForResourceName(file_name, file_type);
        if (!labels_path) {
            LOG(ERROR) << "Failed to find model proto at" << [file_name UTF8String]
            << [file_type UTF8String];
        }
        std::ifstream t;
        t.open([labels_path UTF8String]);
        std::string line;
        while (t) {
            std::getline(t, line);
            label_strings->push_back(line);
        }
        t.close();
    }

    // Returns the top N confidence values over threshold in the provided vector,
    // sorted by confidence in descending order.
    void GetTopN(
                 const float* prediction, const int prediction_size, const int num_results,
                 const float threshold, std::vector<std::pair<float, int> >* top_results) {
        // Will contain top N results in ascending order.
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int> >,
        std::greater<std::pair<float, int> > >
        top_result_pq;

        const long count = prediction_size;
        for (int i = 0; i < count; ++i) {
            const float value = prediction[i];
            // Only add it if it beats the threshold and has a chance at being in
            // the top N.
            if (value < threshold) {
                continue;
            }
            top_result_pq.push(std::pair<float, int>(value, i));

            // If at capacity, kick the smallest value out.
            if (top_result_pq.size() > num_results) {
                top_result_pq.pop();
            }
        }

        // Copy to output vector and reverse into descending order.
        while (!top_result_pq.empty()) {
            top_results->push_back(top_result_pq.top());
            top_result_pq.pop();
        }
        std::reverse(top_results->begin(), top_results->end());
    }
}

@interface ImageViewController()

@end

@implementation ImageViewController {
    std::unique_ptr<tflite::FlatBufferModel> model;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    TfLiteDelegate* delegate;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    
    predictionValues = [[NSMutableDictionary alloc] init];
    NSString* graph_path = FilePathForResourceName(model_file_name, model_file_type);
    model = tflite::FlatBufferModel::BuildFromFile([graph_path UTF8String]);
    if (!model) {
        LOG(FATAL) << "Failed to mmap model " << graph_path;
    }
    LOG(INFO) << "Loaded model " << graph_path;
    model->error_reporter();
    LOG(INFO) << "resolved reporter";
    
    tflite::ops::builtin::BuiltinOpResolver resolver;
    LoadLabels(labels_file_name, labels_file_type, &labels);
    
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    
#if TFLITE_USE_GPU_DELEGATE
    GpuDelegateOptions options;
    options.allow_precision_loss = true;
    options.wait_type = GpuDelegateOptions::WaitType::kActive;
    delegate = NewGpuDelegate(&options);
    interpreter->ModifyGraphWithDelegate(delegate);
#endif
    
    // Explicitly resize the input tensor.
    {
        int input = interpreter->inputs()[0];
        std::vector<int> sizes = {1, wanted_input_width, wanted_input_height, wanted_input_channels};
        //std::vector<int> sizes = {1, 224, 224, 3};
        interpreter->ResizeInputTensor(input, sizes);
    }
    if (!interpreter) {
        LOG(FATAL) << "Failed to construct interpreter";
    }
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LOG(FATAL) << "Failed to allocate tensors!";
    }
}

// Callback function after importImageButton clicked
- (IBAction)importImageButtonClick:(id)sender {
    
    // Create image picker controller.
    UIImagePickerController *imagePickerController = [[UIImagePickerController alloc] init];
    imagePickerController.delegate = self;
    
    // Create actionsheet which provide options either choose from photo library or taken from camera
    UIAlertController *actionSheet = [UIAlertController alertControllerWithTitle:@"Photo Sources" message:@"Please select a source" preferredStyle:UIAlertControllerStyleActionSheet];
    // Camera button
    [actionSheet addAction:[UIAlertAction actionWithTitle:@"Camera" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action) {
        imagePickerController.sourceType = UIImagePickerControllerSourceTypeCamera;
        [self presentViewController:imagePickerController animated:YES completion:nil];
    }]];
    // Photo Library button
    [actionSheet addAction:[UIAlertAction actionWithTitle:@"Photo Library" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action) {
        imagePickerController.sourceType = UIImagePickerControllerSourceTypePhotoLibrary;
        imagePickerController.allowsEditing = false;
        [self presentViewController:imagePickerController animated:YES completion:nil];
    }]];
    // Cancel button
    [actionSheet addAction:[UIAlertAction actionWithTitle:@"Cancel" style:UIAlertActionStyleCancel handler:nil]];
    
    [self presentViewController:actionSheet animated:YES completion:nil];
}

// Load photo taken from camera
- (void) imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary<UIImagePickerControllerInfoKey,id> *)info{
    
    // Correct rotation issue of original photo
    UIImage *image = [info valueForKey:UIImagePickerControllerOriginalImage];
    image = [image fixOrientation];
    
    // Resize the UIImageView to fit the ratio of photo
    double scaledHeight = self.stillImageView.frame.size.width / image.size.width * image.size.height;
    self.stillImageView.frame = CGRectMake(self.stillImageView.frame.origin.x, self.stillImageView.frame.origin.y, self.stillImageView.frame.size.width, scaledHeight);
    self.stillImageView.image = image;
    
    CIImage *ciImage = [[CIImage alloc] initWithImage:self.stillImageView.image];
    
    self.stillImageView.layer.sublayers = nil;
    
    // Detect face
    [self detectFace:ciImage];
    
    // Pornography classification
    image_data imageData = [self CGImageToPixels:image.CGImage];
    [self inputImageToModel:imageData];
    [self runModel];
    
    [self dismissViewControllerAnimated:YES completion:nil];
}

// After imagePicker cancel button pushed
- (void) imagePickerControllerDidCancel:(UIImagePickerController *)picker{
    [self dismissViewControllerAnimated:YES completion:nil];
}

// Draw Face location with bounding box
- (void)drawFaceRect:(CIImage*)image{
    
    // This function only work with iOS and higher
    if (@available(iOS 11.0, *)) {
        VNDetectFaceLandmarksRequest *faceLandmarks = [VNDetectFaceLandmarksRequest new];
        VNSequenceRequestHandler *faceLandmarksDetectionRequest = [VNSequenceRequestHandler new];
        [faceLandmarksDetectionRequest performRequests:@[faceLandmarks] onCIImage:image error:nil];
        
        // Result list of faces detected.
        for(VNFaceObservation *observation in faceLandmarks.results){
            
            //draw rect on face
            CGRect boundingBox = observation.boundingBox;
            
            CGSize size = CGSizeMake(boundingBox.size.width * self.stillImageView.frame.size.width, boundingBox.size.height * self.stillImageView.frame.size.height);
            CGPoint origin = CGPointMake(boundingBox.origin.x * self.stillImageView.frame.size.width, (1-boundingBox.origin.y)*self.stillImageView.frame.size.height - size.height);
            
            CAShapeLayer *layer = [CAShapeLayer layer];
            
            layer.frame = CGRectMake(origin.x, origin.y, size.width, size.height);
            layer.borderColor = [UIColor redColor].CGColor;
            layer.borderWidth = 2;
            
            [self.stillImageView.layer addSublayer:layer];
        }
    } else {
        // Doing nothing
    }
}

// Detect Face from image inputed
- (void)detectFace:(CIImage*)image{
    
    // Only work with iOS 11 or higher
    if (@available(iOS 11.0, *)) {
        VNDetectFaceRectanglesRequest *faceDetectionReq = [VNDetectFaceRectanglesRequest new];
        NSDictionary *d = [[NSDictionary alloc] init];
        //req handler
        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCIImage:image options:d];
        //send req to handler
        [handler performRequests:@[faceDetectionReq] error:nil];
        
        //is there a face?
        for(VNFaceObservation *observation in faceDetectionReq.results){
            if(observation){
                dispatch_async(dispatch_get_main_queue(), ^{
                    [self drawFaceRect:image];
                });
            }
        }
    } else {
        // Doing nothing
    }
}

// Convert CGImage to Pixels to input into TF Lite model.
- (image_data) CGImageToPixels:(CGImage *)image {
    image_data result;
    result.width = (int)CGImageGetWidth(image);
    result.height = (int)CGImageGetHeight(image);
    result.channels = 4;
    
    CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
    const int bytes_per_row = (result.width * result.channels);
    const int bytes_in_image = (bytes_per_row * result.height);
    result.data = std::vector<uint8_t>(bytes_in_image);
    const int bits_per_component = 8;
    
    CGContextRef context =
    CGBitmapContextCreate(result.data.data(), result.width, result.height, bits_per_component, bytes_per_row,
                          color_space, kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(color_space);
    CGContextDrawImage(context, CGRectMake(0, 0, result.width, result.height), image);
    CGContextRelease(context);
    
    return result;
}

// Input Image (as image_data type) into TF Lite Model
- (void)inputImageToModel:(image_data)image{
    float* out = interpreter->typed_input_tensor<float>(0);
   
    assert(image.channels >= wanted_input_channels);
    uint8_t* in = image.data.data();
    
    for (int y = 0; y < wanted_input_height; ++y) {
        const int in_y = (y * image.height) / wanted_input_height;
        uint8_t* in_row = in + (in_y * image.width * image.channels);
        float* out_row = out + (y * wanted_input_width * wanted_input_channels);
        for (int x = 0; x < wanted_input_width; ++x) {
            const int in_x = (x * image.width) / wanted_input_width;
            uint8_t* in_pixel = in_row + (in_x * image.channels);
            float* out_pixel = out_row + (x * wanted_input_channels);
            for (int c = 0; c < wanted_input_channels; ++c) {
                out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
            }
        }
    }
}

// Run TF Lite model.
- (void)runModel {
    double startTimestamp = [[NSDate new] timeIntervalSince1970];
    if (interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke!";
    }
    double endTimestamp = [[NSDate new] timeIntervalSince1970];
    total_latency += (endTimestamp - startTimestamp);
    total_count += 1;
    NSLog(@"Time: %.4lf, avg: %.4lf, count: %d", endTimestamp - startTimestamp,
          total_latency / total_count,  total_count);
    
    const int output_size = (int)labels.size();
    const int kNumResults = 1;
    const float kThreshold = 0.1f;
    
    std::vector<std::pair<float, int>> top_results;
    
    float* output = interpreter->typed_output_tensor<float>(0);
    GetTopN(output, output_size, kNumResults, kThreshold, &top_results);
    
    std::vector<std::pair<float, std::string>> newValues;
    for (const auto& result : top_results) {
        std::pair<float, std::string> item;
        item.first = result.first;
        item.second = labels[result.second];
        
        newValues.push_back(item);
    }
    dispatch_async(dispatch_get_main_queue(), ^(void) {
        [self setPredictionValues:newValues];
    });
}

// Display label and confidnece value on screen
- (void)setPredictionValues:(std::vector<std::pair<float, std::string>>)newValues {
    
        for  (const auto& item : newValues) {
        std::string label = item.second;
        const float value = item.first;
        const int valuePercentage = (int)roundf(value * 100.0f);
        
        NSString* valueText = [NSString stringWithFormat:@"%d%%", valuePercentage];
        
        NSString *nsLabel = [NSString stringWithCString:label.c_str()
                                               encoding:[NSString defaultCStringEncoding]];
            
        self.classificationLabel.text = nsLabel;
        self.percentageLabel.text = valueText;
    }
}

@end
