





#ifndef OCRE
#define OCRE

#include "tesseract/baseapi.h"
#include "leptonica/allheaders.h"
#include "frame.h"

void  ocre_init();
void  ocre_scanImage( IplImage *croppedImage, int digits = 16 );
bool  ocre_complete();
char* ocre_result();
void ocre_reset();
void ocre_destroy();

/* Android 'to_string' std support */
#include <string>
#include <sstream>

template <typename T>
std::string convert_to_string(T value)
{
    std::ostringstream os ;
    os << value ;
    return os.str() ;
}

#endif