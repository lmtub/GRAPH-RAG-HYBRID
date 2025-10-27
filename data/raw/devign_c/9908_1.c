static struct scoop_info_s *spitz_scoop_init(struct pxa2xx_state_s *cpu,

                int count) {

    int iomemtype;

    struct scoop_info_s *s;



    s = (struct scoop_info_s *)

            qemu_mallocz(sizeof(struct scoop_info_s) * 2);

    memset(s, 0, sizeof(struct scoop_info_s) * count);

    s[0].target_base = 0x10800000;

    s[1].target_base = 0x08800040;



    /* Ready */

    s[0].status = 0x02;

    s[1].status = 0x02;



    iomemtype = cpu_register_io_memory(0, scoop_readfn,

                    scoop_writefn, &s[0]);

    cpu_register_physical_memory(s[0].target_base, 0xfff, iomemtype);

    register_savevm("scoop", 0, 0, scoop_save, scoop_load, &s[0]);



    if (count < 2)

        return s;



    iomemtype = cpu_register_io_memory(0, scoop_readfn,

                    scoop_writefn, &s[1]);

    cpu_register_physical_memory(s[1].target_base, 0xfff, iomemtype);

    register_savevm("scoop", 1, 0, scoop_save, scoop_load, &s[1]);



    return s;

}
