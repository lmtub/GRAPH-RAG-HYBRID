bool qemu_log_in_addr_range(uint64_t addr)

{

    if (debug_regions) {

        int i = 0;

        for (i = 0; i < debug_regions->len; i++) {

            Range *range = &g_array_index(debug_regions, Range, i);

            if (addr >= range->begin && addr <= range->end - 1) {

                return true;

            }

        }

        return false;

    } else {

        return true;

    }

}
