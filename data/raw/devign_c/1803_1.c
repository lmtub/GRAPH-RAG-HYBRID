void qemu_system_powerdown(void)

{

    if(pm_state->pmen & PWRBTN_EN) {

        pm_state->pmsts |= PWRBTN_EN;

	pm_update_sci(pm_state);

    }

}
