static int ram_load(QEMUFile *f, void *opaque, int version_id)

{

    int flags = 0, ret = 0, invalid_flags = 0;

    static uint64_t seq_iter;

    int len = 0;

    /*

     * If system is running in postcopy mode, page inserts to host memory must

     * be atomic

     */

    bool postcopy_running = postcopy_state_get() >= POSTCOPY_INCOMING_LISTENING;

    /* ADVISE is earlier, it shows the source has the postcopy capability on */

    bool postcopy_advised = postcopy_state_get() >= POSTCOPY_INCOMING_ADVISE;



    seq_iter++;



    if (version_id != 4) {

        ret = -EINVAL;

    }



    if (!migrate_use_compression()) {

        invalid_flags |= RAM_SAVE_FLAG_COMPRESS_PAGE;

    }

    /* This RCU critical section can be very long running.

     * When RCU reclaims in the code start to become numerous,

     * it will be necessary to reduce the granularity of this

     * critical section.

     */

    rcu_read_lock();



    if (postcopy_running) {

        ret = ram_load_postcopy(f);

    }



    while (!postcopy_running && !ret && !(flags & RAM_SAVE_FLAG_EOS)) {

        ram_addr_t addr, total_ram_bytes;

        void *host = NULL;

        uint8_t ch;



        addr = qemu_get_be64(f);

        flags = addr & ~TARGET_PAGE_MASK;

        addr &= TARGET_PAGE_MASK;



        if (flags & invalid_flags) {

            if (flags & invalid_flags & RAM_SAVE_FLAG_COMPRESS_PAGE) {

                error_report("Received an unexpected compressed page");

            }



            ret = -EINVAL;

            break;

        }



        if (flags & (RAM_SAVE_FLAG_ZERO | RAM_SAVE_FLAG_PAGE |

                     RAM_SAVE_FLAG_COMPRESS_PAGE | RAM_SAVE_FLAG_XBZRLE)) {

            RAMBlock *block = ram_block_from_stream(f, flags);



            host = host_from_ram_block_offset(block, addr);

            if (!host) {

                error_report("Illegal RAM offset " RAM_ADDR_FMT, addr);

                ret = -EINVAL;

                break;

            }

            ramblock_recv_bitmap_set(block, host);

            trace_ram_load_loop(block->idstr, (uint64_t)addr, flags, host);

        }



        switch (flags & ~RAM_SAVE_FLAG_CONTINUE) {

        case RAM_SAVE_FLAG_MEM_SIZE:

            /* Synchronize RAM block list */

            total_ram_bytes = addr;

            while (!ret && total_ram_bytes) {

                RAMBlock *block;

                char id[256];

                ram_addr_t length;



                len = qemu_get_byte(f);

                qemu_get_buffer(f, (uint8_t *)id, len);

                id[len] = 0;

                length = qemu_get_be64(f);



                block = qemu_ram_block_by_name(id);

                if (block) {

                    if (length != block->used_length) {

                        Error *local_err = NULL;



                        ret = qemu_ram_resize(block, length,

                                              &local_err);

                        if (local_err) {

                            error_report_err(local_err);

                        }

                    }

                    /* For postcopy we need to check hugepage sizes match */

                    if (postcopy_advised &&

                        block->page_size != qemu_host_page_size) {

                        uint64_t remote_page_size = qemu_get_be64(f);

                        if (remote_page_size != block->page_size) {

                            error_report("Mismatched RAM page size %s "

                                         "(local) %zd != %" PRId64,

                                         id, block->page_size,

                                         remote_page_size);

                            ret = -EINVAL;

                        }

                    }

                    ram_control_load_hook(f, RAM_CONTROL_BLOCK_REG,

                                          block->idstr);

                } else {

                    error_report("Unknown ramblock \"%s\", cannot "

                                 "accept migration", id);

                    ret = -EINVAL;

                }



                total_ram_bytes -= length;

            }

            break;



        case RAM_SAVE_FLAG_ZERO:

            ch = qemu_get_byte(f);

            ram_handle_compressed(host, ch, TARGET_PAGE_SIZE);

            break;



        case RAM_SAVE_FLAG_PAGE:

            qemu_get_buffer(f, host, TARGET_PAGE_SIZE);

            break;



        case RAM_SAVE_FLAG_COMPRESS_PAGE:

            len = qemu_get_be32(f);

            if (len < 0 || len > compressBound(TARGET_PAGE_SIZE)) {

                error_report("Invalid compressed data length: %d", len);

                ret = -EINVAL;

                break;

            }

            decompress_data_with_multi_threads(f, host, len);

            break;



        case RAM_SAVE_FLAG_XBZRLE:

            if (load_xbzrle(f, addr, host) < 0) {

                error_report("Failed to decompress XBZRLE page at "

                             RAM_ADDR_FMT, addr);

                ret = -EINVAL;

                break;

            }

            break;

        case RAM_SAVE_FLAG_EOS:

            /* normal exit */

            break;

        default:

            if (flags & RAM_SAVE_FLAG_HOOK) {

                ram_control_load_hook(f, RAM_CONTROL_HOOK, NULL);

            } else {

                error_report("Unknown combination of migration flags: %#x",

                             flags);

                ret = -EINVAL;

            }

        }

        if (!ret) {

            ret = qemu_file_get_error(f);

        }

    }



    wait_for_decompress_done();

    rcu_read_unlock();

    trace_ram_load_complete(ret, seq_iter);

    return ret;

}
