static void load_linux(const char *kernel_filename,

		       const char *initrd_filename,

		       const char *kernel_cmdline)

{

    uint16_t protocol;

    uint32_t gpr[8];

    uint16_t seg[6];

    uint16_t real_seg;

    int setup_size, kernel_size, initrd_size, cmdline_size;

    uint32_t initrd_max;

    uint8_t header[1024];

    target_phys_addr_t real_addr, prot_addr, cmdline_addr, initrd_addr;

    FILE *f, *fi;



    /* Align to 16 bytes as a paranoia measure */

    cmdline_size = (strlen(kernel_cmdline)+16) & ~15;



    /* load the kernel header */

    f = fopen(kernel_filename, "rb");

    if (!f || !(kernel_size = get_file_size(f)) ||

	fread(header, 1, 1024, f) != 1024) {

	fprintf(stderr, "qemu: could not load kernel '%s'\n",

		kernel_filename);

	exit(1);

    }



    /* kernel protocol version */

#if 0

    fprintf(stderr, "header magic: %#x\n", ldl_p(header+0x202));

#endif

    if (ldl_p(header+0x202) == 0x53726448)

	protocol = lduw_p(header+0x206);

    else

	protocol = 0;



    if (protocol < 0x200 || !(header[0x211] & 0x01)) {

	/* Low kernel */

	real_addr    = 0x90000;

	cmdline_addr = 0x9a000 - cmdline_size;

	prot_addr    = 0x10000;

    } else if (protocol < 0x202) {

	/* High but ancient kernel */

	real_addr    = 0x90000;

	cmdline_addr = 0x9a000 - cmdline_size;

	prot_addr    = 0x100000;

    } else {

	/* High and recent kernel */

	real_addr    = 0x10000;

	cmdline_addr = 0x20000;

	prot_addr    = 0x100000;

    }



#if 0

    fprintf(stderr,

	    "qemu: real_addr     = 0x" TARGET_FMT_plx "\n"

	    "qemu: cmdline_addr  = 0x" TARGET_FMT_plx "\n"

	    "qemu: prot_addr     = 0x" TARGET_FMT_plx "\n",

	    real_addr,

	    cmdline_addr,

	    prot_addr);

#endif



    /* highest address for loading the initrd */

    if (protocol >= 0x203)

	initrd_max = ldl_p(header+0x22c);

    else

	initrd_max = 0x37ffffff;



    if (initrd_max >= ram_size-ACPI_DATA_SIZE)

	initrd_max = ram_size-ACPI_DATA_SIZE-1;



    /* kernel command line */

    pstrcpy_targphys(cmdline_addr, 4096, kernel_cmdline);



    if (protocol >= 0x202) {

	stl_p(header+0x228, cmdline_addr);

    } else {

	stw_p(header+0x20, 0xA33F);

	stw_p(header+0x22, cmdline_addr-real_addr);

    }



    /* loader type */

    /* High nybble = B reserved for Qemu; low nybble is revision number.

       If this code is substantially changed, you may want to consider

       incrementing the revision. */

    if (protocol >= 0x200)

	header[0x210] = 0xB0;



    /* heap */

    if (protocol >= 0x201) {

	header[0x211] |= 0x80;	/* CAN_USE_HEAP */

	stw_p(header+0x224, cmdline_addr-real_addr-0x200);

    }



    /* load initrd */

    if (initrd_filename) {

	if (protocol < 0x200) {

	    fprintf(stderr, "qemu: linux kernel too old to load a ram disk\n");

	    exit(1);

	}



	fi = fopen(initrd_filename, "rb");

	if (!fi) {

	    fprintf(stderr, "qemu: could not load initial ram disk '%s'\n",

		    initrd_filename);

	    exit(1);

	}



	initrd_size = get_file_size(fi);

	initrd_addr = (initrd_max-initrd_size) & ~4095;



        fprintf(stderr, "qemu: loading initrd (%#x bytes) at 0x" TARGET_FMT_plx

                "\n", initrd_size, initrd_addr);



	if (!fread_targphys_ok(initrd_addr, initrd_size, fi)) {

	    fprintf(stderr, "qemu: read error on initial ram disk '%s'\n",

		    initrd_filename);

	    exit(1);

	}

	fclose(fi);



	stl_p(header+0x218, initrd_addr);

	stl_p(header+0x21c, initrd_size);

    }



    /* store the finalized header and load the rest of the kernel */

    cpu_physical_memory_write(real_addr, header, 1024);



    setup_size = header[0x1f1];

    if (setup_size == 0)

	setup_size = 4;



    setup_size = (setup_size+1)*512;

    kernel_size -= setup_size;	/* Size of protected-mode code */



    if (!fread_targphys_ok(real_addr+1024, setup_size-1024, f) ||

	!fread_targphys_ok(prot_addr, kernel_size, f)) {

	fprintf(stderr, "qemu: read error on kernel '%s'\n",

		kernel_filename);

	exit(1);

    }

    fclose(f);



    /* generate bootsector to set up the initial register state */

    real_seg = real_addr >> 4;

    seg[0] = seg[2] = seg[3] = seg[4] = seg[4] = real_seg;

    seg[1] = real_seg+0x20;	/* CS */

    memset(gpr, 0, sizeof gpr);

    gpr[4] = cmdline_addr-real_addr-16;	/* SP (-16 is paranoia) */



    generate_bootsect(gpr, seg, 0);

}
