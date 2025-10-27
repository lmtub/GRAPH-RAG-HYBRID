static int write_dump_pages(DumpState *s)

{

    int ret = 0;

    DataCache page_desc, page_data;

    size_t len_buf_out, size_out;

#ifdef CONFIG_LZO

    lzo_bytep wrkmem = NULL;

#endif

    uint8_t *buf_out = NULL;

    off_t offset_desc, offset_data;

    PageDescriptor pd, pd_zero;

    uint8_t *buf;

    int endian = s->dump_info.d_endian;

    GuestPhysBlock *block_iter = NULL;

    uint64_t pfn_iter;



    /* get offset of page_desc and page_data in dump file */

    offset_desc = s->offset_page;

    offset_data = offset_desc + sizeof(PageDescriptor) * s->num_dumpable;



    prepare_data_cache(&page_desc, s, offset_desc);

    prepare_data_cache(&page_data, s, offset_data);



    /* prepare buffer to store compressed data */

    len_buf_out = get_len_buf_out(s->page_size, s->flag_compress);

    if (len_buf_out == 0) {

        dump_error(s, "dump: failed to get length of output buffer.\n");

        goto out;

    }



#ifdef CONFIG_LZO

    wrkmem = g_malloc(LZO1X_1_MEM_COMPRESS);

#endif



    buf_out = g_malloc(len_buf_out);



    /*

     * init zero page's page_desc and page_data, because every zero page

     * uses the same page_data

     */

    pd_zero.size = cpu_convert_to_target32(s->page_size, endian);

    pd_zero.flags = cpu_convert_to_target32(0, endian);

    pd_zero.offset = cpu_convert_to_target64(offset_data, endian);

    pd_zero.page_flags = cpu_convert_to_target64(0, endian);

    buf = g_malloc0(s->page_size);

    ret = write_cache(&page_data, buf, s->page_size, false);

    g_free(buf);

    if (ret < 0) {

        dump_error(s, "dump: failed to write page data(zero page).\n");

        goto out;

    }



    offset_data += s->page_size;



    /*

     * dump memory to vmcore page by page. zero page will all be resided in the

     * first page of page section

     */

    while (get_next_page(&block_iter, &pfn_iter, &buf, s)) {

        /* check zero page */

        if (is_zero_page(buf, s->page_size)) {

            ret = write_cache(&page_desc, &pd_zero, sizeof(PageDescriptor),

                              false);

            if (ret < 0) {

                dump_error(s, "dump: failed to write page desc.\n");

                goto out;

            }

        } else {

            /*

             * not zero page, then:

             * 1. compress the page

             * 2. write the compressed page into the cache of page_data

             * 3. get page desc of the compressed page and write it into the

             *    cache of page_desc

             *

             * only one compression format will be used here, for

             * s->flag_compress is set. But when compression fails to work,

             * we fall back to save in plaintext.

             */

             size_out = len_buf_out;

             if ((s->flag_compress & DUMP_DH_COMPRESSED_ZLIB) &&

                    (compress2(buf_out, (uLongf *)&size_out, buf, s->page_size,

                    Z_BEST_SPEED) == Z_OK) && (size_out < s->page_size)) {

                pd.flags = cpu_convert_to_target32(DUMP_DH_COMPRESSED_ZLIB,

                                                   endian);

                pd.size  = cpu_convert_to_target32(size_out, endian);



                ret = write_cache(&page_data, buf_out, size_out, false);

                if (ret < 0) {

                    dump_error(s, "dump: failed to write page data.\n");

                    goto out;

                }

#ifdef CONFIG_LZO

            } else if ((s->flag_compress & DUMP_DH_COMPRESSED_LZO) &&

                    (lzo1x_1_compress(buf, s->page_size, buf_out,

                    (lzo_uint *)&size_out, wrkmem) == LZO_E_OK) &&

                    (size_out < s->page_size)) {

                pd.flags = cpu_convert_to_target32(DUMP_DH_COMPRESSED_LZO,

                                                   endian);

                pd.size  = cpu_convert_to_target32(size_out, endian);



                ret = write_cache(&page_data, buf_out, size_out, false);

                if (ret < 0) {

                    dump_error(s, "dump: failed to write page data.\n");

                    goto out;

                }

#endif

#ifdef CONFIG_SNAPPY

            } else if ((s->flag_compress & DUMP_DH_COMPRESSED_SNAPPY) &&

                    (snappy_compress((char *)buf, s->page_size,

                    (char *)buf_out, &size_out) == SNAPPY_OK) &&

                    (size_out < s->page_size)) {

                pd.flags = cpu_convert_to_target32(

                                        DUMP_DH_COMPRESSED_SNAPPY, endian);

                pd.size  = cpu_convert_to_target32(size_out, endian);



                ret = write_cache(&page_data, buf_out, size_out, false);

                if (ret < 0) {

                    dump_error(s, "dump: failed to write page data.\n");

                    goto out;

                }

#endif

            } else {

                /*

                 * fall back to save in plaintext, size_out should be

                 * assigned to s->page_size

                 */

                pd.flags = cpu_convert_to_target32(0, endian);

                size_out = s->page_size;

                pd.size = cpu_convert_to_target32(size_out, endian);



                ret = write_cache(&page_data, buf, s->page_size, false);

                if (ret < 0) {

                    dump_error(s, "dump: failed to write page data.\n");

                    goto out;

                }

            }



            /* get and write page desc here */

            pd.page_flags = cpu_convert_to_target64(0, endian);

            pd.offset = cpu_convert_to_target64(offset_data, endian);

            offset_data += size_out;



            ret = write_cache(&page_desc, &pd, sizeof(PageDescriptor), false);

            if (ret < 0) {

                dump_error(s, "dump: failed to write page desc.\n");

                goto out;

            }

        }

    }



    ret = write_cache(&page_desc, NULL, 0, true);

    if (ret < 0) {

        dump_error(s, "dump: failed to sync cache for page_desc.\n");

        goto out;

    }

    ret = write_cache(&page_data, NULL, 0, true);

    if (ret < 0) {

        dump_error(s, "dump: failed to sync cache for page_data.\n");

        goto out;

    }



out:

    free_data_cache(&page_desc);

    free_data_cache(&page_data);



#ifdef CONFIG_LZO

    g_free(wrkmem);

#endif



    g_free(buf_out);



    return ret;

}
