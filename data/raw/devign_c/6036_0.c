static void smbios_check_collision(int type, int entry)

{

    if (type < ARRAY_SIZE(first_opt)) {

        if (first_opt[type].seen) {

            if (first_opt[type].headertype != entry) {

                error_report("Can't mix file= and type= for same type");

                loc_push_restore(&first_opt[type].loc);

                error_report("This is the conflicting setting");

                loc_pop(&first_opt[type].loc);

                exit(1);

            }

        } else {

            first_opt[type].seen = true;

            first_opt[type].headertype = entry;

            loc_save(&first_opt[type].loc);

        }

    }

}
