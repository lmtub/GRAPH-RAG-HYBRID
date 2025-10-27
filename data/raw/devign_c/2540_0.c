static void mb_add_mod(MultibootState *s,

                       target_phys_addr_t start, target_phys_addr_t end,

                       target_phys_addr_t cmdline_phys)

{

    char *p;

    assert(s->mb_mods_count < s->mb_mods_avail);



    p = (char *)s->mb_buf + s->offset_mbinfo + MB_MOD_SIZE * s->mb_mods_count;



    stl_p(p + MB_MOD_START,   start);

    stl_p(p + MB_MOD_END,     end);

    stl_p(p + MB_MOD_CMDLINE, cmdline_phys);



    mb_debug("mod%02d: "TARGET_FMT_plx" - "TARGET_FMT_plx"\n",

             s->mb_mods_count, start, end);



    s->mb_mods_count++;

}
