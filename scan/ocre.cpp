
#if COMPILE_DMZ
#include "ocre.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "dmz_debug.h"
#include "dmz_olm.h" /* Luhn Check*/


tesseract::TessBaseAPI *_tessBaseAPI;
char* _cardNumber;

const int NUMDEFS = 7;
const std::string ROTATIONS = "1";
const int DEFINITIONS[7][4][4] = {


  /*laCaixa Visa New*/
  {
    {45, 155, 75, 25},
    {115, 155, 75, 25},
    {195, 155, 75, 25},
    {275, 155, 75, 25}
  },
  /*CBGC Buy Powercard*/
  {
    {15, 256, 45, 20},
    {60, 256, 45, 20},
    {105, 256, 45, 20},
    {150, 256, 45, 20},
  },
  /*Venture Card Laser Personalization Vertical*/
  {
    {20, 78, 70, 23},
    {20, 116, 70, 23},
    {20, 153, 70, 23},
    {20, 192, 70, 23}
  },
  /*CMC World Elite*/
  {
    {18, 108, 56, 20},
    {74, 108, 56, 20},
    {134, 174, 56, 20},
    {194, 174, 56, 20}
  },
  /*Quicksilver Ultragraphics flat print Horizontal*/
  {
    {16, 197, 50, 20},
    {70, 197, 50, 20},
    {122, 197, 50, 20},
    {175, 197, 50, 20}
  },
  /*Capital One All Point 360*/
  {
    {12, 174, 65, 24},
    {75, 174, 65, 24},
    {145, 174, 65, 24},
    {210, 174, 65, 24}
  },
  /*Venture Card Laser Personalization Horizontal*/
  {
    {16, 200, 50, 20},
    {62, 200, 50, 20},
    {112, 200, 50, 20},
    {160, 200, 50, 20}
  }
};

/*
 * Initialize Tesseract API
 * Loads the custom trained data for Capital One Cards.
 */
void ocre_init() {
  _tessBaseAPI = new tesseract::TessBaseAPI();
  if ( _tessBaseAPI->Init(NULL, "co") ) {
     dmz_debug_print("Failed to init Tesseract. Are you missing tessdata?\n");
     _tessBaseAPI = NULL;
  }
}

/**
 * Scan an cropped IplImage
 * param grayscaleCroppedImage the pre-cropped image passed
 *       from card.io in grayscale
 * param digits the number of digits we are looking for (15-16)
 */
void ocre_scanImage( IplImage *grayscaleCroppedImage, int digits) {
  if( grayscaleCroppedImage == NULL || _tessBaseAPI == NULL ) {
    return;
  }
  //convert to something we can manipulate.
  CvMat header, *mat = cvGetMat( grayscaleCroppedImage, &header );
  cv::Mat grayscaleMat= cv::Mat(mat);
  cv::Mat rotatedMat = grayscaleMat.clone();
  //prep rotated matrix
  cv::transpose(rotatedMat, rotatedMat);
  cv::flip(rotatedMat, rotatedMat, 1);
  std::string numberWithoutSpaces = "";
  uint8_t * luhnCheck = (uint8_t*)malloc(sizeof(uint8_t)*digits);

  _tessBaseAPI->SetImage(grayscaleMat.data, grayscaleMat.cols, grayscaleMat.rows, grayscaleMat.channels(), grayscaleMat.step1());
  _tessBaseAPI->SetVariable("tessedit_char_whitelist", "0123456789");
  
  for(int i = 0; i<NUMDEFS; i++) {
    bool needsRotation = ROTATIONS.find(convert_to_string(i)) != std::string::npos;
    if( needsRotation ) {
       _tessBaseAPI->SetImage(rotatedMat.data, rotatedMat.cols, rotatedMat.rows, rotatedMat.channels(), rotatedMat.step1());
    }
    for(int j = 0; j<4; j++) {
        _tessBaseAPI->SetRectangle(DEFINITIONS[i][j][0], DEFINITIONS[i][j][1], DEFINITIONS[i][j][2], DEFINITIONS[i][j][3]);
        _tessBaseAPI->Recognize(0);
        numberWithoutSpaces.append(_tessBaseAPI->GetUTF8Text());
    }
    if( needsRotation ) {
      //reset the image
      _tessBaseAPI->SetImage(grayscaleMat.data, grayscaleMat.cols, grayscaleMat.rows, grayscaleMat.channels(), grayscaleMat.step1());
    }

    numberWithoutSpaces.erase(remove_if(numberWithoutSpaces.begin(), numberWithoutSpaces.end(), (int(*)(int)) isspace), numberWithoutSpaces.end());
  
    if ( numberWithoutSpaces.length() != digits)
    {
      numberWithoutSpaces = "";
      continue;
    }
    
    for(int i=0; i<digits; i++) {
      luhnCheck[i] = atoi(numberWithoutSpaces.substr(i,1).c_str());
    }

    if( dmz_passes_luhn_checksum(luhnCheck, digits) )  {
      ocre_reset();
      _cardNumber = (char*)numberWithoutSpaces.c_str();
      free(luhnCheck);
      return;
    }
    numberWithoutSpaces = "";

  }
  free(luhnCheck);
}

/**
 * Checks if the scan is complete
 * returns true if a valid card number was detected
 */
bool ocre_complete() {
  if( _cardNumber == NULL || _cardNumber == "") {
    return false;
  }
  return true;
}

/**
 * Returns the current detected card number
 */
char* ocre_result() {
  return _cardNumber;
}

/*
 * Resets the current card number for another scan
 */
void ocre_reset() {
  _cardNumber = "";
}

/*
 * Release API
 */
void ocre_destroy() {
    if( _tessBaseAPI == NULL ){
        return;
    }
    _tessBaseAPI->End();
}

#endif