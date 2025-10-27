static uint64_t icp_pit_read(void *opaque, hwaddr offset,

                             unsigned size)

{

    icp_pit_state *s = (icp_pit_state *)opaque;

    int n;



    /* ??? Don't know the PrimeCell ID for this device.  */

    n = offset >> 8;

    if (n > 2) {

        qemu_log_mask(LOG_GUEST_ERROR, "%s: Bad timer %d\n", __func__, n);


    }



    return arm_timer_read(s->timer[n], offset & 0xff);

}