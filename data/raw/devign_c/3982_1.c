static int ram_load_postcopy(QEMUFile *f)

{

    int flags = 0, ret = 0;

    bool place_needed = false;

    bool matching_page_sizes = qemu_host_page_size == TARGET_PAGE_SIZE;

    MigrationIncomingState *mis = migration_incoming_get_current();

    /* Temporary page that is later 'placed' */

    void *postcopy_host_page = postcopy_get_tmp_page(mis);

    void *last_host = NULL;



    while (!ret && !(flags & RAM_SAVE_FLAG_EOS)) {

        ram_addr_t addr;

        void *host = NULL;

        void *page_buffer = NULL;

        void *place_source = NULL;

        uint8_t ch;

        bool all_zero = false;



        addr = qemu_get_be64(f);

        flags = addr & ~TARGET_PAGE_MASK;

        addr &= TARGET_PAGE_MASK;



        trace_ram_load_postcopy_loop((uint64_t)addr, flags);

        place_needed = false;

        if (flags & (RAM_SAVE_FLAG_COMPRESS | RAM_SAVE_FLAG_PAGE)) {

            host = host_from_stream_offset(f, addr, flags);

            if (!host) {

                error_report("Illegal RAM offset " RAM_ADDR_FMT, addr);

                ret = -EINVAL;

                break;

            }

            page_buffer = host;

            /*

             * Postcopy requires that we place whole host pages atomically.

             * To make it atomic, the data is read into a temporary page

             * that's moved into place later.

             * The migration protocol uses,  possibly smaller, target-pages

             * however the source ensures it always sends all the components

             * of a host page in order.

             */

            page_buffer = postcopy_host_page +

                          ((uintptr_t)host & ~qemu_host_page_mask);

            /* If all TP are zero then we can optimise the place */

            if (!((uintptr_t)host & ~qemu_host_page_mask)) {

                all_zero = true;

            } else {

                /* not the 1st TP within the HP */

                if (host != (last_host + TARGET_PAGE_SIZE)) {

                    error_report("Non-sequential target page %p/%p\n",

                                  host, last_host);

                    ret = -EINVAL;

                    break;

                }

            }





            /*

             * If it's the last part of a host page then we place the host

             * page

             */

            place_needed = (((uintptr_t)host + TARGET_PAGE_SIZE) &

                                     ~qemu_host_page_mask) == 0;

            place_source = postcopy_host_page;

        }




        switch (flags & ~RAM_SAVE_FLAG_CONTINUE) {

        case RAM_SAVE_FLAG_COMPRESS:

            ch = qemu_get_byte(f);

            memset(page_buffer, ch, TARGET_PAGE_SIZE);

            if (ch) {

                all_zero = false;

            }

            break;



        case RAM_SAVE_FLAG_PAGE:

            all_zero = false;

            if (!place_needed || !matching_page_sizes) {

                qemu_get_buffer(f, page_buffer, TARGET_PAGE_SIZE);

            } else {

                /* Avoids the qemu_file copy during postcopy, which is

                 * going to do a copy later; can only do it when we

                 * do this read in one go (matching page sizes)

                 */

                qemu_get_buffer_in_place(f, (uint8_t **)&place_source,

                                         TARGET_PAGE_SIZE);

            }

            break;

        case RAM_SAVE_FLAG_EOS:

            /* normal exit */

            break;

        default:

            error_report("Unknown combination of migration flags: %#x"

                         " (postcopy mode)", flags);

            ret = -EINVAL;

        }



        if (place_needed) {

            /* This gets called at the last target page in the host page */

            if (all_zero) {

                ret = postcopy_place_page_zero(mis,

                                               host + TARGET_PAGE_SIZE -

                                               qemu_host_page_size);

            } else {

                ret = postcopy_place_page(mis, host + TARGET_PAGE_SIZE -

                                               qemu_host_page_size,

                                               place_source);

            }

        }

        if (!ret) {

            ret = qemu_file_get_error(f);

        }

    }



    return ret;

}