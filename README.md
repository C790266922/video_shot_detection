## Video shot detection

1.Color histogram

```
python3 main.py --hist <threshold>
```

![](./result/result_hist_4.png)

2.Moment invariants

```
python3 main.py --moment <threshold>
```

![](./result/result_mom_0.00001.png)

3.Intersect

```
python3 main.py --mix 2 --hist <threshold1> --moment <threshold2>
```

![](./result/result_intersect.png)

4.Union

```
python3 main.py --mix 1 --hist <threshold1> --moment <threshold2>
```

![](./result/result_union.png)

