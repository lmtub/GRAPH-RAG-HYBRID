kern_return_t FindEjectableCDMedia( io_iterator_t *mediaIterator )

{

    kern_return_t       kernResult;

    mach_port_t     masterPort;

    CFMutableDictionaryRef  classesToMatch;



    kernResult = IOMasterPort( MACH_PORT_NULL, &masterPort );

    if ( KERN_SUCCESS != kernResult ) {

        printf( "IOMasterPort returned %d\n", kernResult );

    }



    classesToMatch = IOServiceMatching( kIOCDMediaClass );

    if ( classesToMatch == NULL ) {

        printf( "IOServiceMatching returned a NULL dictionary.\n" );

    } else {

    CFDictionarySetValue( classesToMatch, CFSTR( kIOMediaEjectableKey ), kCFBooleanTrue );

    }

    kernResult = IOServiceGetMatchingServices( masterPort, classesToMatch, mediaIterator );

    if ( KERN_SUCCESS != kernResult )

    {

        printf( "IOServiceGetMatchingServices returned %d\n", kernResult );

    }



    return kernResult;

}
