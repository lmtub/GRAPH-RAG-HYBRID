void pc_system_firmware_init(MemoryRegion *rom_memory)

{

    DriveInfo *pflash_drv;

    PcSysFwDevice *sysfw_dev;



    /*

     * TODO This device exists only so that users can switch between

     * use of flash and ROM for the BIOS.  The ability to switch was

     * created because flash doesn't work with KVM.  Once it does, we

     * should drop this device for new machine types.

     */

    sysfw_dev = (PcSysFwDevice*) qdev_create(NULL, "pc-sysfw");



    qdev_init_nofail(DEVICE(sysfw_dev));



    if (sysfw_dev->rom_only) {

        old_pc_system_rom_init(rom_memory);

        return;

    }



    pflash_drv = drive_get(IF_PFLASH, 0, 0);



    /* Currently KVM cannot execute from device memory.

       Use old rom based firmware initialization for KVM. */

    /*

     * This is a Bad Idea, because it makes enabling/disabling KVM

     * guest-visible.  Do it only in bug-compatibility mode.

     */

    if (pc_sysfw_flash_vs_rom_bug_compatible && kvm_enabled()) {

        if (pflash_drv != NULL) {

            fprintf(stderr, "qemu: pflash cannot be used with kvm enabled\n");

            exit(1);

        } else {

            sysfw_dev->rom_only = 1;

            old_pc_system_rom_init(rom_memory);

            return;

        }

    }



    /* If a pflash drive is not found, then create one using

       the bios filename. */

    if (pflash_drv == NULL) {

        pc_fw_add_pflash_drv();

        pflash_drv = drive_get(IF_PFLASH, 0, 0);

    }



    if (pflash_drv != NULL) {

        pc_system_flash_init(rom_memory, pflash_drv);

    } else {

        fprintf(stderr, "qemu: PC system firmware (pflash) not available\n");

        exit(1);

    }

}
