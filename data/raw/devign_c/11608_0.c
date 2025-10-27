static AHCIQState *ahci_boot(void)

{

    AHCIQState *s;

    const char *cli;



    s = g_malloc0(sizeof(AHCIQState));



    cli = "-drive if=none,id=drive0,file=%s,cache=writeback,serial=%s"

        ",format=qcow2"

        " -M q35 "

        "-device ide-hd,drive=drive0 "

        "-global ide-hd.ver=%s";

    s->parent = qtest_pc_boot(cli, tmp_path, "testdisk", "version");

    alloc_set_flags(s->parent->alloc, ALLOC_LEAK_ASSERT);



    /* Verify that we have an AHCI device present. */

    s->dev = get_ahci_device(&s->fingerprint);



    return s;

}
