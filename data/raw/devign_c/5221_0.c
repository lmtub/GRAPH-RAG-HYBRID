void pc_cmos_set_s3_resume(void *opaque, int irq, int level)

{

    ISADevice *s = opaque;



    if (level) {

        rtc_set_memory(s, 0xF, 0xFE);

    }

}
