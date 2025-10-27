void do_info_roms(Monitor *mon)

{

    Rom *rom;



    QTAILQ_FOREACH(rom, &roms, next) {

        if (!rom->fw_file) {

            monitor_printf(mon, "addr=" TARGET_FMT_plx

                           " size=0x%06zx mem=%s name=\"%s\" \n",

                           rom->addr, rom->romsize,

                           rom->isrom ? "rom" : "ram",

                           rom->name);

        } else {

            monitor_printf(mon, "fw=%s%s%s"

                           " size=0x%06zx name=\"%s\" \n",

                           rom->fw_dir ? rom->fw_dir : "",

                           rom->fw_dir ? "/" : "",

                           rom->fw_file,

                           rom->romsize,

                           rom->name);

        }

    }

}
