BlockDriverState *bdrv_find_backing_image(BlockDriverState *bs,

        const char *backing_file)

{

    char *filename_full = NULL;

    char *backing_file_full = NULL;

    char *filename_tmp = NULL;

    int is_protocol = 0;

    BlockDriverState *curr_bs = NULL;

    BlockDriverState *retval = NULL;



    if (!bs || !bs->drv || !backing_file) {

        return NULL;

    }



    filename_full     = g_malloc(PATH_MAX);

    backing_file_full = g_malloc(PATH_MAX);

    filename_tmp      = g_malloc(PATH_MAX);



    is_protocol = path_has_protocol(backing_file);



    for (curr_bs = bs; curr_bs->backing_hd; curr_bs = curr_bs->backing_hd) {



        /* If either of the filename paths is actually a protocol, then

         * compare unmodified paths; otherwise make paths relative */

        if (is_protocol || path_has_protocol(curr_bs->backing_file)) {

            if (strcmp(backing_file, curr_bs->backing_file) == 0) {

                retval = curr_bs->backing_hd;

                break;

            }

        } else {

            /* If not an absolute filename path, make it relative to the current

             * image's filename path */

            path_combine(filename_tmp, PATH_MAX, curr_bs->filename,

                         backing_file);



            /* We are going to compare absolute pathnames */

            if (!realpath(filename_tmp, filename_full)) {

                continue;

            }



            /* We need to make sure the backing filename we are comparing against

             * is relative to the current image filename (or absolute) */

            path_combine(filename_tmp, PATH_MAX, curr_bs->filename,

                         curr_bs->backing_file);



            if (!realpath(filename_tmp, backing_file_full)) {

                continue;

            }



            if (strcmp(backing_file_full, filename_full) == 0) {

                retval = curr_bs->backing_hd;

                break;

            }

        }

    }



    g_free(filename_full);

    g_free(backing_file_full);

    g_free(filename_tmp);

    return retval;

}
