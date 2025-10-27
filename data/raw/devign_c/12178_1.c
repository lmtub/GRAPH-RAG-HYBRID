static void spapr_rtc_class_init(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);



    dc->realize = spapr_rtc_realize;

    dc->vmsd = &vmstate_spapr_rtc;





    spapr_rtas_register(RTAS_GET_TIME_OF_DAY, "get-time-of-day",

                        rtas_get_time_of_day);

    spapr_rtas_register(RTAS_SET_TIME_OF_DAY, "set-time-of-day",

                        rtas_set_time_of_day);

}