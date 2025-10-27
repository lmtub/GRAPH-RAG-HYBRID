kern_return_t GetBSDPath( io_iterator_t mediaIterator, char *bsdPath, CFIndex maxPathSize )

{

    io_object_t     nextMedia;

    kern_return_t   kernResult = KERN_FAILURE;

    *bsdPath = '\0';

    nextMedia = IOIteratorNext( mediaIterator );

    if ( nextMedia )

    {

        CFTypeRef   bsdPathAsCFString;

    bsdPathAsCFString = IORegistryEntryCreateCFProperty( nextMedia, CFSTR( kIOBSDNameKey ), kCFAllocatorDefault, 0 );

        if ( bsdPathAsCFString ) {

            size_t devPathLength;

            strcpy( bsdPath, _PATH_DEV );

            strcat( bsdPath, "r" );

            devPathLength = strlen( bsdPath );

            if ( CFStringGetCString( bsdPathAsCFString, bsdPath + devPathLength, maxPathSize - devPathLength, kCFStringEncodingASCII ) ) {

                kernResult = KERN_SUCCESS;

            }

            CFRelease( bsdPathAsCFString );

        }

        IOObjectRelease( nextMedia );

    }



    return kernResult;

}
