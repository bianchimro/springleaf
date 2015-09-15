import pandas
import os
import sys
import random
import base64
import StringIO
import sklearn
from toolz import itertoolz
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
"""
- analyze column types

"""

def analyze_training(target, out, target_column):
    print "loading dataset .."
    df = pandas.read_csv(target, low_memory=False, na_values = ['NA', 'NaN'])
    print df.describe()

    #removing target column
    target_data = df.pop(target_column)

    dtypes_map = get_dtypes_map(df)
    #print dtypes_map

    ids_columns = detect_ids_columns(df, dtypes_map)
    constants_columns = get_constants_columns(df, dtypes_map)
    #print constants_columns

    #let's clean it up
    cc = []
    col_types = {}
    for x in constants_columns:
        print constants_columns[x]
        col_types[x] = [a for a in dtypes_map[x] if a not in constants_columns[x] and a not in ids_columns]
        cc.extend(col_types[x])

    num_cols = []
    for x in col_types:
        print x

        if x != 'object':
            num_cols.extend(col_types[x])

    obj_cols = [x for x in cc if x not in num_cols]

    obj_df = df[obj_cols]
    print obj_df.describe()
    
    num_df = df[num_cols]
    
    print num_df.describe()
    
    num_df.dropna(axis=1, how='all')
    num_df.dropna()

    print num_df.describe()
    return

    #get our compacted dataset
    variate_df = df[cc]
    
    #dropping nas on columns
    variate_df = variate_df.dropna(axis=1, how='all')
    print variate_df.describe()

    #dropping nas on rows
    #varvariate_df.dropna(subset=num_cols)
    
    #print variate_df.describe()

    #print variate_df

    return


    dfs = drop_nas(variate_df, col_types)
    
    initial = True
    final_df = None

    for x in dfs:
        if initial:
            final_df = dfs[x]
        else:
            final_df = final_df.join(dfs[x])
        

    
    print final_df.columns

    


def drop_nas(df, col_types, nas=['NA', 'NaN'], nas_map={}):
    #let's drop nas
    #separate datesets
    by_type = {}
    out = {}
    for x in col_types:
        by_type[x] = df[col_types[x]]
        nas_t = [a for a in nas]
        if x in nas_map:
            nas_t.extend(nas_map[x])
        xx = by_type[x]
        

        cond = False
        for col in xx.columns:  
            cond = cond | (xx[col].all() not in nas_t)
        #xx = xx.drop(cond)
        #if x != 'object':
        #    xx = xx.dropna()
        out[x] = xx
    return out


def get_constants_columns(df, dtypes_map):
    out = {}
    for x in dtypes_map:
        dft = df[dtypes_map[x]]
        o = dft.describe()
        if x == 'object':
            constants_columns = [a for a in o if o[a]['unique']==1]
        else:
            constants_columns = [a for a in o if o[a]['min']==o[a]['max']]

        out[x] = constants_columns

    return out
            

def detect_ids_columns(df, dtypes_map):
    df2 = df.astype(basestring)
    o = df2.describe()
    ids_columns = [a for a in o if o[a]['count']==o[a]['unique']]
    return ids_columns




def get_dtypes_map(df):
    out = {}
    dtypes = []
    for c in df.columns:
        dtypes.append(df[c].dtype)

    dtypes = list(set(dtypes))
    
    for x in dtypes:

        columns = [c for c in df.columns if df[c].dtype == x]
        out[str(x)] = columns        
    return out









def do_analyze2(target, out, TARGET_COLUMN=-1):
    print "loading dataset .."
    df = pandas.read_csv(target)


    
    print "computing descriptive stats .."
    o = df.describe()
    print o

    return

    descr_html = StringIO.StringIO()
    for cols in itertoolz.partition(6, df.columns):
        descr_html.write(o.to_html(columns=cols, float_format='{0:.5f}'.format, classes=["table", "table-striped", "table-condensed"] ))

    
    with open(out, "wb") as of:
        of.write(descr_html)

    return
    with open(out, "wb") as of:
        of.write(o)

    return

    print "finding constants .."
    ok_std = [x for x in o if o[x]['min']!=o[x]['max']]
    ko_std = [x for x in o if x not in ok_std]

    constants_html = StringIO.StringIO()
    for x in ko_std:
        constants_html.write("<p>%s</p>" % str(x))


    
    with open(out, "ab") as of:
        of.write(constants_html)    

    return





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


from sklearn import tree
def do_model(target, target_column):

    print "loading dataset .."
    df = pandas.read_csv(target, low_memory=False)

    target_data = df.pop(target_column)

    dtypes_map = get_dtypes_map(df)
    #print dtypes_map

    ids_columns = detect_ids_columns(df, dtypes_map)
    constants_columns = get_constants_columns(df, dtypes_map)
    #print constants_columns

    #let's clean it up
    cc = []
    col_types = {}
    for x in constants_columns:
        print constants_columns[x]
        col_types[x] = [a for a in dtypes_map[x] if a not in constants_columns[x] and a not in ids_columns]
        cc.extend(col_types[x])

    num_cols = []
    for x in col_types:
        print x

        if x != 'object':
            num_cols.extend(col_types[x])

    obj_cols = [x for x in cc if x not in num_cols]

    obj_df = df[obj_cols]
    #print obj_df.describe()
    
    num_df = df[num_cols]
    
    #print num_df.describe()
    
    num_df.dropna(axis=1, how='all')
    num_df.dropna()

    
    variate_df = df[cc]
    
    #dropping nas on columns
    variate_df = variate_df.dropna(axis=1, how='all')
    #print variate_df.describe()
    #return
    X = variate_df[num_cols[:80]]
    print X.shape

    clf = ExtraTreesClassifier()
    X_new = clf.fit(X, target_data).transform(X)
    print clf.feature_importances_  

    print X_new.shape



    print "onto modeling.."
    #some_cols = num_cols[:60]
    #X = variate_df[num_cols[:10]]

    #estimator = tree.DecisionTreeClassifier()

    estimator = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
    
    score = sklearn.cross_validation.cross_val_score(estimator, 
        X_new, y=target_data, scoring='roc_auc', cv=5, n_jobs=1, verbose=0, fit_params=None)
    
    print score
    return

    #estimator = estimator.fit(X, target_data)
    #print estimator.tree_

    #sklearn.tree.export_graphviz(estimator, out_file='tree.dot') 




    """
    target_true = [True for x in target_data]
    target_false = [False for x in target_data]

    print 'roc_auc_score dummy', roc_auc_score(target_data, target_data)
    print 'roc_auc_score true', roc_auc_score(target_data, target_true)
    print 'roc_auc_score false', roc_auc_score(target_data, target_false)
    """


if __name__ == '__main__':
    target = sys.argv[1]

    if "--analyze" in sys.argv:
        out_file = sys.argv[2]
        analyze_training(target, out_file, 'target')

    if "--model" in sys.argv:
        do_model(target, 'target')






