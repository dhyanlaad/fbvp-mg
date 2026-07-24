.libPaths(c("~/R/x86_64-pc-linux-gnu-library/4.3", .libPaths()))
library(MetricGraph)

# Define Tadpole graph geometry
edge1 <- rbind(c(0, 0), c(1, 0))              # e0: length 1.0
edge2 <- rbind(c(1, 0), c(0.5, sqrt(3)/2))    # e1: length 1.0
edge3 <- rbind(c(0.5, sqrt(3)/2), c(0, 0))    # e2: length 1.0
edge4 <- rbind(c(0, 0), c(0, -1.3))           # e3: length 1.3

edges <- list(edge1, edge2, edge3, edge4)
graph <- metric_graph$new(edges = edges)

# Generate points on each edge
L_vec <- c(1.0, 1.0, 1.0, 1.3)
pts_per_edge <- 500
PtE <- do.call(rbind, lapply(1:4, function(i) {
  cbind(rep(i, pts_per_edge), seq(0, L_vec[i], length.out = pts_per_edge))
}))

# Parameter setup (matching the Python scale, roughly)
# nu=1.5 translates to alpha=2
set.seed(42)
u <- sample_spde(range = 1.5, sigma = 1.0, alpha = 2, graph = graph, PtE = PtE)

# Save to CSV
df <- data.frame(edge = PtE[,1] - 1, distance = PtE[,2], value = as.numeric(u))
write.csv(df, "matern_field.csv", row.names = FALSE)
cat("Successfully generated matern_field.csv\n")
