import cv2

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes, Details
from msrest.authentication import CognitiveServicesCredentials

import azure.ai.vision as sdk


key = '361dd87102f7494c8829b5e26840bed4'
endPoint = 'https://con-bib-cogsvc-compvision.cognitiveservices.azure.com/'

def image_analysis(imagepath: str):
    service_options = sdk.VisionServiceOptions(endPoint, key)
    vision_source = sdk.VisionSource(filename=imagepath)

    analysis_options = sdk.ImageAnalysisOptions()

    analysis_options.features = (
        sdk.ImageAnalysisFeature.TEXT |
        sdk.ImageAnalysisFeature.TAGS
    )


    image_analyser = sdk.ImageAnalyzer(service_options, vision_source, analysis_options)

    print()
    print('please wait for results')

    lines = []
    tagNames = []
    tagConfifence = []
    

    result = image_analyser.analyze()

    if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:

        if result.tags is not None: 
            print("Tags: ")
            for tag in result.tags:
                print(" '{}', confidence {:.4f}".format(tag.name, tag.confidence))
                tagNames.append(tag.name)
                tagConfifence.append(tag.confidence)



        
        if result.text is not None: 
            print(" Text:")
            for line in result.text.lines:
                points_string = "{" + ", ".join([str(int(point)) for point in line.bounding_polygon]) + "}"
                print("   Line: '{}', Bounding polygon {}".format(line.content, points_string))
                lines.append(line.content)
                
                ##print(line)
                for word in line.words:
                    points_string = "{" + ", ".join([str(int(point)) for point in word.bounding_polygon]) + "}"
                    print("     Word: '{}', Bounding polygon {}, Confidence {:.4f}"
                          .format(word.content, points_string, word.confidence))
                    
                    

                    

        
        result_details = sdk.ImageAnalysisResultDetails.from_result(result)
        print(" Result details:")
        print("   Image ID: {}".format(result_details.image_id))
        print("   Result ID: {}".format(result_details.result_id))
        print("   Connection URL: {}".format(result_details.connection_url))
        print("   JSON result: {}".format(result_details.json_result))
                    
    else:

        error_details = sdk.ImageAnalysisErrorDetails.from_result(result)
        print(" Analysis failed.")
        print("   Error reason: {}".format(error_details.reason))
        print("   Error code: {}".format(error_details.error_code))
        print("   Error message: {}".format(error_details.message))
        print(" Did you set the computer vision endpoint and key?")


    

    tagDictionary = dict(zip(tagNames, tagConfifence))

    return {'Tags': tagDictionary,
            'VRN': lines}





data = image_analysis("./Res/red_car.jpg")

print()
print()
print() 

print(data)