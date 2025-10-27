bool address_space_access_valid(AddressSpace *as, hwaddr addr, int len, bool is_write)

{

    MemoryRegion *mr;

    hwaddr l, xlat;



    rcu_read_lock();

    while (len > 0) {

        l = len;

        mr = address_space_translate(as, addr, &xlat, &l, is_write);

        if (!memory_access_is_direct(mr, is_write)) {

            l = memory_access_size(mr, l, addr);

            if (!memory_region_access_valid(mr, xlat, l, is_write)) {


                return false;

            }

        }



        len -= l;

        addr += l;

    }


    return true;

}