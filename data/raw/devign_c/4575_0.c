static int rwpipe_read_ppm_header( rwpipe *rw, int *width, int *height )

{

    char line[ 3 ];

    FILE *in = rwpipe_reader( rw );

    int max;



    fgets( line, 3, in );

    if ( !strncmp( line, "P6", 2 ) )

    {

        *width = rwpipe_read_number( rw );

        *height = rwpipe_read_number( rw );

        max = rwpipe_read_number( rw );

        return max != 255 || *width <= 0 || *height <= 0;

    }

    return 1;

}
