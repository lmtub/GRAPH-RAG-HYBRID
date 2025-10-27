static void read_guest_mem(void)

{

    uint32_t *guest_mem;

    gint64 end_time;

    int i, j;

    size_t size;



    g_mutex_lock(data_mutex);



    end_time = g_get_monotonic_time() + 5 * G_TIME_SPAN_SECOND;

    while (!fds_num) {

        if (!_cond_wait_until(data_cond, data_mutex, end_time)) {

            /* timeout has passed */

            g_assert(fds_num);

            break;

        }

    }



    /* check for sanity */

    g_assert_cmpint(fds_num, >, 0);

    g_assert_cmpint(fds_num, ==, memory.nregions);



    /* iterate all regions */

    for (i = 0; i < fds_num; i++) {



        /* We'll check only the region statring at 0x0*/

        if (memory.regions[i].guest_phys_addr != 0x0) {

            continue;

        }



        g_assert_cmpint(memory.regions[i].memory_size, >, 1024);



        size =  memory.regions[i].memory_size + memory.regions[i].mmap_offset;



        guest_mem = mmap(0, size, PROT_READ | PROT_WRITE,

                         MAP_SHARED, fds[i], 0);



        g_assert(guest_mem != MAP_FAILED);

        guest_mem += (memory.regions[i].mmap_offset / sizeof(*guest_mem));



        for (j = 0; j < 256; j++) {

            uint32_t a = readl(memory.regions[i].guest_phys_addr + j*4);

            uint32_t b = guest_mem[j];



            g_assert_cmpint(a, ==, b);

        }



        munmap(guest_mem, memory.regions[i].memory_size);

    }



    g_assert_cmpint(1, ==, 1);

    g_mutex_unlock(data_mutex);

}
