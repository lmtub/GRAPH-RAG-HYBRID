static int pick_formats(AVFilterGraph *graph)

{

    int i, j, ret;

    int change;



    do{

        change = 0;

        for (i = 0; i < graph->filter_count; i++) {

            AVFilterContext *filter = graph->filters[i];

            if (filter->nb_inputs){

                for (j = 0; j < filter->nb_inputs; j++){

                    if(filter->inputs[j]->in_formats && filter->inputs[j]->in_formats->format_count == 1) {

                        pick_format(filter->inputs[j], NULL);

                        change = 1;

                    }

                }

            }

            if (filter->nb_outputs){

                for (j = 0; j < filter->nb_outputs; j++){

                    if(filter->outputs[j]->in_formats && filter->outputs[j]->in_formats->format_count == 1) {

                        pick_format(filter->outputs[j], NULL);

                        change = 1;

                    }

                }

            }

            if (filter->nb_inputs && filter->nb_outputs && filter->inputs[0]->format>=0) {

                for (j = 0; j < filter->nb_outputs; j++) {

                    if(filter->outputs[j]->format<0) {

                        pick_format(filter->outputs[j], filter->inputs[0]);

                        change = 1;

                    }

                }

            }

        }

    }while(change);



    for (i = 0; i < graph->filter_count; i++) {

        AVFilterContext *filter = graph->filters[i];



        for (j = 0; j < filter->nb_inputs; j++)

            if ((ret = pick_format(filter->inputs[j], NULL)) < 0)

                return ret;

        for (j = 0; j < filter->nb_outputs; j++)

            if ((ret = pick_format(filter->outputs[j], NULL)) < 0)

                return ret;

    }

    return 0;

}
