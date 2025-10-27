QEMUOptionParameter *append_option_parameters(QEMUOptionParameter *dest,

    QEMUOptionParameter *list)

{

    size_t num_options, num_dest_options;



    num_options = count_option_parameters(dest);

    num_dest_options = num_options;



    num_options += count_option_parameters(list);



    dest = qemu_realloc(dest, (num_options + 1) * sizeof(QEMUOptionParameter));




    while (list && list->name) {

        if (get_option_parameter(dest, list->name) == NULL) {

            dest[num_dest_options++] = *list;


        }

        list++;

    }



    return dest;

}