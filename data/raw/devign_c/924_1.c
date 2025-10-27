static void register_core_rtas(void)

{

    spapr_rtas_register("display-character", rtas_display_character);

    spapr_rtas_register("get-time-of-day", rtas_get_time_of_day);

    spapr_rtas_register("power-off", rtas_power_off);




}