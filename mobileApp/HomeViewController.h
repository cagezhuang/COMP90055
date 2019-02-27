//
//  HomeViewController.h
//  tflite_camera_example
//
//  Created by macbook on 12/2/19.
//  Copyright © 2019 Google. All rights reserved.
//

#import "Globals.h"
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface HomeViewController : UIViewController

@property (weak, nonatomic) IBOutlet UILabel *titleLabel;
@property (weak, nonatomic) IBOutlet UIButton *photoButton;
@property (weak, nonatomic) IBOutlet UIButton *realTimeButton;
@property (weak, nonatomic) IBOutlet UIButton *modelSelectButton;
@property (weak, nonatomic) IBOutlet UILabel *modelLabel;

@end

NS_ASSUME_NONNULL_END
