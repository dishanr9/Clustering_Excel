
import core
import view
import timeit

start = timeit.default_timer()
customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139', '#FF6633', '#e8f0ba', '#8c6f8d', '#dddead', '#66eeee', '#59aec7']

# try:
data_chunks = core.read_data()
distances = []
ntop = 500
print("len : ", len(data_chunks))
length = len(data_chunks)
loops = (length) + (length - 2 if length % 2 == 0 else length - 3)
print("Loops: \t", loops)


i = 0
m = []
while 1 + (4 * i) <= loops:
    m.append(1+(4*i))
    i = i + 1

i = 0
while 2 + (4 * i) <= loops:
    m.append(2+(4*i))
    i = i + 1

j = 0
# for i in range(loops):
#
#     if i in m or i == 0:
#         distances = core.clustering(data_chunks[j], j)
#
#         if not distances.empty:
#
#             df_entropy = core.build_comparison_matrix(distances, j, top=5)
#
#             if not df_entropy.empty:
#                 df_ntop = df_entropy[df_entropy["DeltaEntropy%"] < -48][["Dataset1", "Dataset2", "DeltaEntropy%","Distance"]]
#                 df_ntop = df_ntop.sort_values(by=["Distance"], ascending=True).head(ntop)
#
#                 d1 = df_ntop[df_ntop.duplicated("Dataset1", keep="first")]
#                 df_ntop = df_ntop[~df_ntop.isin(d1)].dropna(how="all")
#
#                 d2 = df_ntop[df_ntop.duplicated("Dataset2", keep="first")]
#                 df_ntop = df_ntop[~df_ntop.isin(d2)].dropna(how="all")
#
#                 core.merge(df_entropy, distances, df_ntop, j)
#                 core.time_trace("t"+str(i))
#                 view.export_excel(i)
#
#         j = j + 1
#     else:
#         core.consolidate_plot(i)
#         # core.consolidate_plot(i+1)
#
# # core.consolidate_plot(i+1)
# # core.consolidate_plot(i+2)
# i = i + 1
c = 1
i = 0
j = 1

for k in range(int(length / 2)):
    while c <= 2:
        print("HEY", j)
        if i >= length:
            c = 3
            break

        distances = core.clustering(data_chunks[i], i)

        if not distances.empty:
            df_entropy = core.build_comparison_matrix(distances, i, top=5)

            if not df_entropy.empty:
                # Don't take anything less than 48% reduction
                # -41<-40
                df_ntop = df_entropy[df_entropy["DeltaEntropy%"] < -61][
                    ["Dataset1", "Dataset2", "DeltaEntropy%", "Distance"]]
                df_ntop = df_ntop.sort_values(by=["Distance"], ascending=True).head(ntop)

                d1 = df_ntop[df_ntop.duplicated("Dataset1", keep="first")]
                df_ntop = df_ntop[~df_ntop.isin(d1)].dropna(how="all")

                d2 = df_ntop[df_ntop.duplicated("Dataset2", keep="first")]
                df_ntop = df_ntop[~df_ntop.isin(d2)].dropna(how="all")

                core.merge(df_entropy, distances, df_ntop, i)
                view.export_excel(i)

        if i != 0:
            c = c + 1
        i = i + 1
    #     checking git push

    while c > 1:
        core.consolidate_plot(j)
        print("Consolidate_plot{0}".format(j))
        c = c - 1
        j = j + 1

c = True
m = 1
while c:
    df_entropy_cons = core.consolidate()
    df_ntop_cons = core.check_consolidation(j, df_entropy_cons)
    print("df_ntop_cons:",df_ntop_cons)
    if df_ntop_cons.empty:
        c = False
    else:
        core.consolidate_model(j, df_ntop_cons, df_entropy_cons)
        j = j + 1
        m = m + 1

# alt_data, ground_name = core.gmm_fulldata()

# if alt_data is not None:
#     core.build_heatmap_matrix(alt_data, ground_name)
#
# core.time_trace("merge_5")

# except FileNotFoundError as e:
#     print("File doesn't exist!")

stop = timeit.default_timer()

print('Time: ', stop - start)