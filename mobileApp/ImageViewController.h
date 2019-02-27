//
//  ImageViewController.h
//  AIR
//
//  Created by macbook on 12/2/19.
//  Copyright © 2019 Google. All rights reserved.
//

#import "UIImage+fixOrientation.h"
#import "Globals.h"
#import <AVFoundation/AVFoundation.h>
#import <UIKit/UIKit.h>
#import <Vision/Vision.h>
#import <Photos/Photos.h>

#include <vector>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"

NS_ASSUME_NONNULL_BEGIN

typedef struct {
    int width;
    int height;
    int channels;
    std::vector<uint8_t> data;
} image_data;


@interface ImageViewController : UIViewController <UINavigationControllerDelegate, UIImagePickerControllerDelegate> {
    NSMutableDictionary *predictionValues;
    
    std::vector<std::string> labels;
    double total_latency;
    int total_count;
}
@property (weak, nonatomic) IBOutlet UIImageView *stillImageView;
@property (weak, nonatomic) IBOutlet UILabel *classificationLabel;
@property (weak, nonatomic) IBOutlet UILabel *percentageLabel;
@property (weak, nonatomic) IBOutlet UIButton *importImageButton;

@end

NS_ASSUME_NONNULL_END
