import pandas
from pandas.tools.plotting import bootstrap_plot
from pandas.tools.plotting import scatter_matrix
import os
import matplotlib.pyplot as plt
import sys
from toolz import itertoolz
import  numpy as np
from pandas.tools.plotting import parallel_coordinates
import StringIO
import spectra
from sklearn.cross_validation import cross_val_predict

from sklearn_pandas import DataFrameMapper, cross_val_score
import sklearn.preprocessing, sklearn.decomposition, sklearn.linear_model, \
         sklearn.pipeline, sklearn.metrics

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

import random
import base64


scalepos = spectra.scale([ "#fff", "teal" ])
scaleneg = spectra.scale([ "#fff", "crimson" ])

def fmt(x):
    if(x) > 0:
        col = scalepos(x).hexcode
    else:
        col = scaleneg(abs(x)).hexcode
    return "<div style='border-bottom:solid 2px %s'></div>%.02f" % (col, x)

def get_corr_formatter(df):
    out = {}
    for x in df.columns:
        out[x] = fmt
    return out



def do_analyze(target, out, TARGET_COLUMN=-1):
    with open("report_template.html") as tpl:
        template = tpl.read()   

    print "loading dataset .."
    df = pandas.read_csv(target, delim_whitespace=True, header=None, na_values=[-1], dtype='float64')
    df = df.dropna()

    print "computing descriptive stats .."
    o = df.describe(percentiles=[.05, .10, .25, .50, .75, .90, .95, .98])
    descr_html = StringIO.StringIO()
    for cols in itertoolz.partition(6, df.columns):
        descr_html.write(o.to_html(columns=cols, float_format='{0:.5f}'.format, classes=["table", "table-striped", "table-condensed"] ))

    print "finding constants .."
    ok_std = [x for x in o if o[x]['min']!=o[x]['max']]
    ko_std = [x for x in o if x not in ok_std]

    constants_html = StringIO.StringIO()
    for x in ko_std:
        constants_html.write("<p>%s</p>" % str(x))


    clean_df = df[ok_std]
    attrs = [x for x in clean_df.columns if x!=TARGET_COLUMN]


    print "computing correlation .."
    corr_spearman_html = StringIO.StringIO()
    corr_spearman = clean_df.corr(method="spearman")
    corr_spearman_html.write(corr_spearman.to_html(float_format='{0:.5f}'.format, classes=["table", "correlation"],
        escape=False ))

    print corr_spearman

    return

    color_script = """
    <script>
    $(function(){

        var scale = d3.scale.linear()
          .range(["red", "#fff", "teal"])
          .domain([-1.0, 0, 1.0]);
      
        $('.correlation td').each(function(n, it){
          var i = $(it);
          var val = parseFloat(i.text());
          i.css('background-color', scale(val));
        });

    });

    </script>
    """


    

    done_cols = { }
    attrs = [x for x in clean_df.columns if x != TARGET_COLUMN]

    imgtags = []
    for x in attrs:
        clean_df.plot(kind='scatter', x=x, y=TARGET_COLUMN, alpha=.25, c='skyblue')    
        plt.savefig("scatter_%d.png" % x)

        with open("scatter_%d.png" % x) as fl:
            pngdata = base64.b64encode(fl.read())
            pngdatauri = "data:image/png;base64,"+pngdata
            imgtags.append("<div><img src='%s'></div>" % pngdatauri)

    scatter_plots = ""
    for im in imgtags:
        scatter_plots += im


    out_html = template.format(descriptive=descr_html.getvalue(), constants=constants_html.getvalue(),
        corr_spearman=corr_spearman_html.getvalue(), filename=target, color_script=color_script,
        scatter_plots = scatter_plots )

    #out = "describe-dataset.html"
    with open(out, "wb") as outfile:
        outfile.write(out_html)



    def do_model(target, pca_components = 8, TARGET_COLUMN=-1):

        print "loading dataset .."
        df = pandas.read_csv(target, delim_whitespace=True, header=None, na_values=[-1], dtype='float64')
        df = df.dropna()

        print "computing descriptive stats .."
        o = df.describe(percentiles=[.05, .10, .25, .50, .75, .90, .95, .98])

        print "finding constants .."
        ok_std = [x for x in o if o[x]['min']!=o[x]['max']]

        print "doing models ..."

        clean_df = df[ok_std]
        attrs = [x for x in clean_df.columns if x!=TARGET_COLUMN]

        scale = lambda x: (x - x.mean()) / x.std()
        clean_df_scaled = clean_df.apply(scale)

        print "doing pca on scaled dataset"

        pca = sklearn.decomposition.PCA(n_components=pca_components)
        new_x = pca.fit_transform(clean_df_scaled[attrs])
        print "pca.explained_variance_ratio_:"
        print(pca.explained_variance_ratio_) 

        df_x = pandas.DataFrame(new_x)
        df_y = clean_df_scaled[TARGET_COLUMN]

        #print df_x.describe()

        rgr = SVR(verbose=3)
        #rgr =  DecisionTreeRegressor(max_depth=10)

        predicted = cross_val_predict(rgr, df_x, df_y, cv=10)
        fig,ax = plt.subplots()
        ax.scatter(df_y, predicted)
        ax.plot([df_y.min(), df_y.max()], [df_y.min(), df_y.max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')

        plt.savefig("cv_predictions.png")


        """
        pipe = sklearn.pipeline.Pipeline([
            #('lm', sklearn.linear_model.LinearRegression()),
            ('lm', DecisionTreeRegressor(max_depth=5))

            ])
        """

        #print cross_val_score(pipe, clean_df_scaled[attrs], clean_df_scaled[6], 'mean_absolute_error')


        """
        for x in range(5):
            
            test_df = some(clean_df_scaled, 100000)
            

            #lr = SVR(kernel='linear', C=1e3)
            predicted = cross_val_predict(lr, test_df[attrs], test_df[6], cv=5)

            fig,ax = plt.subplots()
            ax.scatter(test_df[6], predicted)
            ax.plot([test_df[6].min(), test_df[6].max()], [test_df[6].min(), test_df[6].max()], 'k--', lw=4)
            ax.set_xlabel('Measured')
            ax.set_ylabel('Predicted')

            plt.savefig("xxx_%d.png" % x)

        sys.exit(0)
        """

        """
        print "computing covariances .."
        covariances_html = StringIO.StringIO()
        covariances =  clean_df.cov()
        covariances_html.write(covariances.to_html(float_format='{0:.5f}'.format, classes=["table"] ))
        """




















#target = "../data/midLife/Endurance_SET3_RRdattone_3.txt"


if __name__ == '__main__':
    target = sys.argv[1]

    if "--analyze" in sys.argv:
        out_file = sys.argv[2]
        do_analyze(target, out_file)

    if "--model" in sys.argv:
        do_model(target)






