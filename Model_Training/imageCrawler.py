
"""
 AUTHOR : Ren Lei
 PURPOSE : Crawl photos from Google Image Search
"""

import os
from datetime import date
from icrawler.builtin import GoogleImageCrawler

TEST_IMAGE_DIR =  os.getcwd() + '/test_images'
NON_PORN_FOLDER = 'non_porn'
PORN_FOLDER = 'porn'

# Define subfolder for non-porn folder. The search should match the name in the list
NON_PORN_LIST = [
    'baby',
    'boxing',
    'architecture',
    'landscape',
    'portrait',
    'swimmer',
    'wrestling',
    'beach girl',
    'sumo wrestling',
    'massage'
]
# Define subfolder for porn folder
PORN_LIST = [
    'porn'
]

# Try to create sub-folder. If exists, create next one.
try:
    os.mkdir(TEST_IMAGE_DIR + '/' + NON_PORN_FOLDER)
    print('Directory: ', NON_PORN_FOLDER, ' created')
except FileExistsError:
    print('Directory: ', NON_PORN_FOLDER, ' already exists')
try:
    os.mkdir(TEST_IMAGE_DIR + '/' + PORN_FOLDER)
    print('Directory: ', PORN_FOLDER, ' created')
except FileExistsError:
    print('Directory: ', PORN_FOLDER, ' already exists')

# Iteration of non-porn list untill all photos are fetched
for c, name in enumerate(NON_PORN_LIST):
    folderDir = TEST_IMAGE_DIR + '/' + NON_PORN_FOLDER + '/' + name
    try:
        os.mkdir(folderDir)
        print('Directory: ', name, ' created')
    except FileExistsError:
        print('Directory: ', name, ' already exists')
        continue

    google_crawler = GoogleImageCrawler(
        parser_threads=2,
        downloader_threads=4,storage={'root_dir': folderDir})
    google_crawler.crawl(keyword=name, max_num=100)

# Iteration of porn list untill all photos are fetched
for c, name in enumerate(PORN_LIST):
    folderDir = TEST_IMAGE_DIR + '/' + PORN_FOLDER + '/' + name
    try:
        os.mkdir(folderDir)
        print('Directory: ', name, ' created')
    except FileExistsError:
        print('Directory: ', name, ' already exists')
        continue

    google_crawler = GoogleImageCrawler(
        parser_threads=2,
        downloader_threads=4, storage={'root_dir': folderDir})
    google_crawler.crawl(keyword=name, max_num=500)