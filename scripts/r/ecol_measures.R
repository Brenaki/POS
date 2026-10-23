# ECoL complexity measures for every CSV bag in a directory (ADR 0023).
#
# Usage: Rscript ecol_measures.R <ecol_pkg_dir> <in_dir> <out_csv> <groups>
#
# <ecol_pkg_dir> is an *extracted CRAN source tarball* of ECoL. Its R/ files
# are sourced into a private environment instead of `library(ECoL)`, so two
# versions (0.3.0, the one the legacy Cpx.py was written for, and 0.4.4) can
# be compared side by side without installing either. Only N1 and the network
# group call igraph; they come out NA when igraph is not installed.
# Each CSV holds the features followed by a last `class` column.

args <- commandArgs(trailingOnly=TRUE)
ecol <- new.env()
for (f in sort(list.files(file.path(args[1], "R"), full.names=TRUE))) {
  sys.source(f, envir=ecol)
}
groups <- strsplit(args[4], ",")[[1]]
has_igraph <- requireNamespace("igraph", quietly=TRUE)

run_group <- function(g, x, y) {
  fn <- get(paste0(g, ".default"), envir=ecol)
  ms <- get(paste0("ls.", g), envir=ecol)()
  vals <- setNames(rep(NA_real_, length(ms)), ms)
  for (m in ms) {
    if (!has_igraph && (m == "N1" || g == "network")) next
    set.seed(0)  # N4 interpolates random points
    v <- tryCatch(unlist(fn(x, y, measures=m, summary="mean"))[1],
                  error=function(e) NA_real_)
    vals[m] <- as.numeric(v)
  }
  vals
}

rows <- list()
for (f in sort(list.files(args[2], pattern="\\.csv$", full.names=TRUE))) {
  d <- read.csv(f)
  x <- d[, -ncol(d), drop=FALSE]
  y <- d[, ncol(d)]
  for (g in groups) {
    t0 <- proc.time()[["elapsed"]]
    vals <- run_group(g, x, y)
    secs <- proc.time()[["elapsed"]] - t0
    rows[[length(rows) + 1]] <- data.frame(
      file=basename(f), group=g, measure=names(vals), value=unname(vals),
      seconds=secs / length(vals))
  }
}
write.csv(do.call(rbind, rows), args[3], row.names=FALSE)
