AVFilter *avfilter_get_by_name(const char *name)

{

    int i;



    for (i = 0; registered_avfilters[i]; i++)

        if (!strcmp(registered_avfilters[i]->name, name))

            return registered_avfilters[i];



    return NULL;

}
