---
title: "DS-Projects"
permalink: /ds-projects/
layout: splash
author_profile: false
classes: wide
---

<div class="projects-container">

<div class="project-card" id="faa-birdstrike-analysis">
  <h2>FAA Bird Strike Analysis: Database Design & Statistical Analysis</h2>
  
  <div class="project-metadata">
    <span class="project-tech"><i class="fab fa-r-project"></i> R, MySQL, dplyr, ggplot2, RMySQL, knitr</span>
    <a href="https://github.com/rishipat160/FAABirdStrikeAnalysis" class="project-link"><i class="fab fa-github"></i> View Code</a>
  </div>

  <details>
    <summary><strong>Project Overview</strong></summary>
    <div class="project-details">
      <p>In this project, I designed and implemented a relational database to analyze wildlife strikes to aircraft using FAA data. The project involved data modeling, database creation, data cleaning, and statistical analysis to identify patterns and trends in bird strikes.</p>
      
      <h3>Database Design</h3>
      <p>I designed a normalized database schema with four main tables:</p>
      <ul>
        <li><strong>Airports:</strong> Information about airport locations and identifiers</li>
        <li><strong>Flights:</strong> Details about aircraft, airlines, and flight dates</li>
        <li><strong>Conditions:</strong> Weather and sky conditions during incidents</li>
        <li><strong>Incidents:</strong> Specific details about wildlife strikes including altitude and impact</li>
      </ul>
      
      <p>The schema included appropriate primary and foreign keys to maintain referential integrity:</p>
      
      {% highlight sql %}
-- Airports Table
CREATE TABLE IF NOT EXISTS airports (
  aid INT,
  airportName TEXT,
  airportState TEXT,
  airportCode TEXT,
  PRIMARY KEY (aid)
);

-- Flights Table
CREATE TABLE IF NOT EXISTS flights (
  fid INT,
  date DATE,
  originAirport INT,
  airlineName TEXT,
  aircraftType TEXT,
  isHeavy BOOLEAN,
  PRIMARY KEY (fid),
  FOREIGN KEY (originAirport) REFERENCES airports(aid)
);

-- Conditions Table
CREATE TABLE IF NOT EXISTS conditions (
  cid INT,
  sky_condition TEXT,
  explanation TEXT,
  PRIMARY KEY (cid)
);

-- Incidents Table
CREATE TABLE IF NOT EXISTS incidents (
  iid INT,
  fid INT,
  wlsize TEXT,
  impact TEXT,
  altitude INT CHECK (altitude >= 0),
  conditions INT,
  PRIMARY KEY (iid),
  FOREIGN KEY (fid) REFERENCES flights(fid),
  FOREIGN KEY (conditions) REFERENCES conditions(cid)
);
      {% endhighlight %}
      
      <h3>Data Cleaning and Transformation</h3>
      <p>I implemented several data cleaning functions to handle issues in the raw dataset:</p>
      <ul>
        <li>Converting date strings to standardized SQL date format</li>
        <li>Handling missing values with appropriate defaults</li>
        <li>Cleaning altitude data by removing commas and validating numeric values</li>
        <li>Creating unique identifiers for airports and conditions</li>
      </ul>
      
      <p>Example of the date cleaning function:</p>
      
      {% highlight r %}
# Clean Date function
clean_date <- function(date_string) {
  if (date_string == "") {
    return(NA)
  } 
  else {
    cleaned = substr(date_string, 1, nchar(date_string) - 5)
    converted_date = as.Date(cleaned, format="%m/%d/%y")
    return(format(converted_date, "%Y-%m-%d"))
  }
}
      {% endhighlight %}
      
      <h3>Database Implementation</h3>
      <p>I used RMySQL to connect to an Amazon RDS MySQL instance, create the schema, and load the cleaned data:</p>
      
      {% highlight r %}
# Creating a connection to Amazon RDS
mysqldb <- dbConnect(RMySQL::MySQL(), 
                    user = db_user, 
                    password = db_password,
                    dbname = db_name,
                    host = db_host, 
                    port = db_port)

# Create tables and load data
dbSendQuery(mysqldb, createAirports)
dbSendQuery(mysqldb, createFlights)
dbSendQuery(mysqldb, createConditions)
dbSendQuery(mysqldb, createIncidents)

# Write dataframes to database tables
dbWriteTable(mysqldb, "airports", df.airports, append = TRUE, row.names = FALSE)
dbWriteTable(mysqldb, "conditions", df.conditions, append = TRUE, row.names = FALSE)
dbWriteTable(mysqldb, "flights", df.flights, append = TRUE, row.names = FALSE)
dbWriteTable(mysqldb, "incidents", df.incidents, append = TRUE, row.names = FALSE)
      {% endhighlight %}
      
      <h3>Analysis and Findings</h3>
      <p>I performed several analytical queries to extract insights from the data:</p>
      
      <h4>Top Airlines with Bird Strikes</h4>
      <p>I identified which airlines experienced the most wildlife strikes:</p>
      
      {% highlight sql %}
SELECT airlineName, COUNT(*) as numOfStrikes 
FROM flights 
JOIN incidents on incidents.fid = flights.fid 
GROUP BY airlineName 
ORDER BY numOfStrikes desc 
LIMIT 5
      {% endhighlight %}
      
      <img src="/assets/images/topairstrike.png" alt="Top Airlines with Bird Strikes" class="project-image">
      
      <h4>Airports with Above-Average Incidents</h4>
      <p>I used a Common Table Expression (CTE) to find airports with higher than average strike incidents:</p>
      
      {% highlight sql %}
WITH airport_strikes AS (
  SELECT airports.airportName, COUNT(*) as numOfStrikes 
  FROM airports 
  JOIN flights ON airports.aid = flights.originAirport 
  JOIN incidents ON flights.fid = incidents.fid 
  GROUP BY airports.airportName
), 
average_incidents as (
  SELECT AVG(numOfStrikes) as averageNumIncidents 
  FROM airport_strikes
)
SELECT airport_strikes.airportName, airport_strikes.numOfStrikes 
FROM airport_strikes, average_incidents 
WHERE airport_strikes.numOfStrikes > average_incidents.averageNumIncidents 
ORDER BY numOfStrikes desc 
LIMIT 5
      {% endhighlight %}
      
      <img src="/assets/images/airportanalysis.png" alt="Airports with Above-Average Incidents" class="project-image">
      
      <h4>Yearly Trend Analysis</h4>
      <p>I analyzed the trend of wildlife strikes over time:</p>
      
      {% highlight sql %}
SELECT YEAR(date) as year, COUNT(*) as numOfStrikes 
FROM flights 
GROUP BY YEAR(date) 
ORDER BY year
      {% endhighlight %}
      
      <img src="/assets/images/analysisyear.png" alt="Yearly Trend Analysis" class="project-image">
      
      <p>I visualized this data using R's plotting capabilities to identify long-term trends:</p>
      
      {% highlight r %}
plot(yearlyStrikes$year, yearlyStrikes$numOfStrikes,
     type = "l",
     main = "Wildlife Strikes by Year",
     xlab = "Year",
     ylab = "Number of Strikes")
points(yearlyStrikes$year, yearlyStrikes$numOfStrikes, pch = 4)
grid()
      {% endhighlight %}
      
      <img src="/assets/images/trendbyyear.png" alt="Trend of Bird Strikes by Year" class="project-image">
      
      <h3>Stored Procedure Implementation</h3>
      <p>I created a stored procedure to update incident records while maintaining an audit log:</p>
      
      {% highlight sql %}
CREATE PROCEDURE update_incident(
    IN p_iid INT,
    IN new_altitude INT,
    IN new_impact TEXT,
    IN new_wlsize TEXT
)
BEGIN
    -- Creates Table if needed
    CREATE TABLE IF NOT EXISTS updateLog (
        uid INT AUTO_INCREMENT PRIMARY KEY,
        modification_type TEXT,
        tableName TEXT,
        time DATETIME,
        original_altitude INT,
        original_impact TEXT,
        original_wlsize TEXT
    );
    
    -- Updates Log
    INSERT INTO updateLog (modification_type, tableName, time, original_altitude, original_impact, original_wlsize)
    SELECT 'updating', 'incident', NOW(), altitude, impact, wlsize
    FROM incidents
    WHERE iid = p_iid;
    
    -- Updates actual incident
    UPDATE incidents
    SET altitude = new_altitude,
        impact = new_impact,
        wlsize = new_wlsize
    WHERE iid = p_iid;
END
      {% endhighlight %}
      
      <h3>Key Findings</h3>
      <p>The analysis revealed several important insights:</p>
      <ul>
        <li>Wildlife strikes have been steadily increasing over the years, possibly due to better reporting</li>
        <li>Certain airports have significantly higher incident rates, likely due to their geographic location</li>
        <li>Major airlines with more flights naturally experience more strikes, but the rate per flight varies</li>
        <li>Weather conditions play a significant role in strike frequency</li>
      </ul>
      
      <h3>Technologies Used</h3>
      <ul>
        <li>R for data processing and analysis</li>
        <li>MySQL for database management</li>
        <li>RMySQL for database connectivity</li>
        <li>knitr and kableExtra for report generation</li>
        <li>Amazon RDS for cloud database hosting</li>
      </ul>
    </div>
  </details>
</div>

<div class="project-card" id="pharma-sales-db">
  <h2>Pharmaceutical Sales Database: Analytics & Visualization Platform</h2>
  
  <div class="project-metadata">
    <span class="project-tech"><i class="fab fa-r-project"></i> R, MySQL, XML, dplyr, ggplot2, RMySQL, RSQLite</span>
    <a href="https://github.com/rishipat160/PharmaDataWarehouse" class="project-link"><i class="fab fa-github"></i> View Code</a>
  </div>

  <details>
    <summary><strong>Project Overview</strong></summary>
    <div class="project-details">
      <p>In this project, I designed and implemented a comprehensive pharmaceutical sales data warehouse and analytics platform. The project involved extracting data from XML sources, creating a normalized relational database, building a star schema for analytics, and developing visualizations to derive business insights.</p>
      
      <h3>Data Processing Pipeline</h3>
      <p>I developed a complete ETL (Extract, Transform, Load) pipeline using R to process pharmaceutical sales data:</p>
      <ul>
        <li>Extracted sales transaction data from multiple XML files</li>
        <li>Transformed and cleaned the data to ensure consistency</li>
        <li>Loaded the data into a normalized SQLite database</li>
        <li>Created a star schema in MySQL for analytical processing</li>
      </ul>
      
      <h3>Database Design</h3>
      <p>The project involved two database designs:</p>
      
      <h4>Normalized Relational Database</h4>
      <p>I designed a normalized database schema with four main tables:</p>
      <ul>
        <li><strong>Reps:</strong> Sales representatives information</li>
        <li><strong>Products:</strong> Pharmaceutical product details</li>
        <li><strong>Customers:</strong> Customer information with location data</li>
        <li><strong>Sales:</strong> Transaction records with appropriate foreign keys</li>
      </ul>
      
      <p>The schema included appropriate primary and foreign keys to maintain referential integrity:</p>
      
      {% highlight sql %}
-- Reps Table
CREATE TABLE reps (
  rep_id INTEGER PRIMARY KEY,
  last_name TEXT,
  first_name TEXT,
  phone TEXT,
  hire_date TEXT,
  commission TEXT,
  territory TEXT,
  certified BOOLEAN
);

-- Products Table
CREATE TABLE products (
  product_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT UNIQUE,
  unit_cost REAL,
  currency TEXT
);

-- Customers Table
CREATE TABLE customers (
  customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT,
  country TEXT,
  UNIQUE(name, country)
);

-- Sales Table
CREATE TABLE sales (
  sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
  rep_id INTEGER,
  customer_id INTEGER,
  product_id INTEGER,
  date TEXT,
  quantity INTEGER,
  FOREIGN KEY(rep_id) REFERENCES reps(rep_id),
  FOREIGN KEY(customer_id) REFERENCES customers(customer_id),
  FOREIGN KEY(product_id) REFERENCES products(product_id)
);
      {% endhighlight %}
      
      <h4>Star Schema for Analytics</h4>
      <p>I implemented a star schema in MySQL to facilitate efficient analytical queries:</p>
      <ul>
        <li><strong>Sales Facts:</strong> Central fact table with sales metrics and dimensional keys</li>
        <li><strong>Rep Facts:</strong> Aggregated sales performance by representative</li>
        <li><strong>Time Dimension:</strong> Hierarchical time attributes (year, quarter, month)</li>
        <li><strong>Product Dimension:</strong> Product attributes and pricing information</li>
        <li><strong>Geography Dimension:</strong> Customer location information</li>
      </ul>
      
      {% highlight sql %}
-- Sales Facts Table
CREATE TABLE sales_facts (
  sale_id INT AUTO_INCREMENT PRIMARY KEY,
  period VARCHAR(7),
  sale_date DATE,
  month INT,
  quarter INT,
  year INT,
  product_name TEXT,
  country_name TEXT,
  total_amount DECIMAL(10, 2),
  total_units INT,
  product_unit_price DECIMAL(10, 2),
  product_currency TEXT
);

-- Rep Facts Table
CREATE TABLE rep_facts (
  rep_id_facts INT AUTO_INCREMENT PRIMARY KEY,
  rep_id INT,
  rep_name TEXT,
  month INT,
  quarter INT,
  year INT,
  total_amount DECIMAL(10, 2),
  average_amount DECIMAL(10, 2)
);
      {% endhighlight %}
      
      <h3>Data Extraction and Transformation</h3>
      <p>I implemented several custom parsing functions to extract and transform the XML data:</p>
      
      {% highlight r %}
# Parse XML sales node
parse_sales_node <- function(node) {
  rep_id <- as.integer(xmlAttrs(node)["repID"])
  date <- parse_date(xmlValue(node[["sale"]][["date"]]))
  product_name <- xmlValue(node[["sale"]][["product"]])
  quantity <- as.integer(xmlValue(node[["sale"]][["qty"]]))
  customer_name <- xmlValue(node[["customer"]])
  
  # Retrieve product_id and customer_id from the mapping
  product_id <- product_id_map[product_name]
  customer_id <- customer_id_map[customer_name]
  
  return(data.frame(
    rep_id = rep_id,
    customer_id = customer_id,
    product_id = product_id,
    date = as.character(date),
    quantity = quantity,
    stringsAsFactors = FALSE
  ))
}
      {% endhighlight %}
      
      <h3>Star Schema Population</h3>
      <p>I developed a process to transform the normalized data into the star schema for analytics:</p>
      
      {% highlight r %}
# Loop through years and months to populate the star schema
for (year in years$year) {
  for (month in months) {
    ensureConnected(mysqldb)
    data_query <- sprintf("
      SELECT 
        strftime('%%Y-%%m', s.date) AS period,
        p.name AS product_name,
        s.date AS sale_date,
        c.country AS country_name,
        strftime('%%m', s.date) AS month, 
        CASE 
          WHEN strftime('%%m', s.date) BETWEEN '01' AND '03' THEN 1
          WHEN strftime('%%m', s.date) BETWEEN '04' AND '06' THEN 2
          WHEN strftime('%%m', s.date) BETWEEN '07' AND '09' THEN 3
          WHEN strftime('%%m', s.date) BETWEEN '10' AND '12' THEN 4
        END AS quarter,
        strftime('%%Y', s.date) AS year, 
        SUM(s.quantity) AS total_units,
        SUM(s.quantity * p.unit_cost) AS total_amount,
        p.unit_cost AS product_unit_price,
        p.currency AS product_currency
      FROM 
        sales s
      JOIN 
        products p ON s.product_id = p.product_id
      JOIN 
        customers c ON s.customer_id = c.customer_id
      WHERE 
        strftime('%%Y', s.date) = '%s' AND strftime('%%m', s.date) = '%02d'
      GROUP BY 
        p.name, c.country;
    ", year, month)
    
    monthly_data <- dbGetQuery(sqlite_db, data_query)
    
    if (nrow(monthly_data) > 0) {
      dbWriteTable(mysqldb, 'sales_facts', monthly_data, append = TRUE, row.names = FALSE)
    }
  }
}
      {% endhighlight %}
      
      <h3>Analysis and Findings</h3>
      <p>I performed several analytical queries to extract business insights from the data:</p>
      
      <h4>Top Products by Revenue</h4>
      <p>I identified the pharmaceutical products generating the highest revenue:</p>
      
      {% highlight sql %}
SELECT product_name, SUM(total_amount) AS total_revenue
FROM sales_facts
GROUP BY product_name
ORDER BY total_revenue DESC
LIMIT 5;
      {% endhighlight %}
      
      <p>Results:</p>
      <pre>
  product_name total_revenue
1      Zalofen    90,179,700
2 Bhiktarvizem     9,442,800
3   Xinoprozen     6,786,612
4   Xipralofen     5,777,688
5  Proxinostat     5,337,951
      </pre>
      
      <h4>Quarterly Product Performance</h4>
      <p>I analyzed product performance by quarter to identify seasonal trends:</p>
      
      {% highlight sql %}
SELECT product_name, quarter, year, SUM(total_amount) AS total_revenue, 
       SUM(total_units) AS total_units_sold
FROM sales_facts
GROUP BY product_name, year, quarter
ORDER BY product_name, year, quarter;
      {% endhighlight %}
      
      <p>Sample results for one product:</p>
      <pre>
  product_name quarter year total_revenue total_units_sold
1  Alaraphosol       1 2020        102120           111000
2  Alaraphosol       2 2020        105248           114400
3  Alaraphosol       3 2020        112792           122600
4  Alaraphosol       4 2020         80868            87900
5  Alaraphosol       1 2021        193844           210700
6  Alaraphosol       2 2021        210404           228700
7  Alaraphosol       3 2021        209760           228000
      </pre>
      
      <h4>Sales by Country</h4>
      <p>I analyzed sales distribution across different countries to identify key markets:</p>
      
      <img src="/assets/images/revenuebycountry.png" alt="Revenue by Country" class="project-image">
      
      <h4>Sales Representative Performance</h4>
      <p>I tracked sales performance by representative across quarters to identify top performers:</p>
      
      <img src="/assets/images/salesperrepperquarter.png" alt="Sales per Rep per Quarter" class="project-image">
      
      <h3>Key Findings</h3>
      <p>The analysis revealed several important business insights:</p>
      <ul>
        <li>Zalofen is the highest-revenue product, generating over 90 million in sales</li>
        <li>Sales show seasonal patterns with Q2 and Q3 typically showing stronger performance</li>
        <li>Certain markets (countries) contribute disproportionately to overall revenue</li>
        <li>Sales representative performance varies significantly, with top performers generating up to 3x the revenue of average performers</li>
      </ul>
      
      <h3>Technologies Used</h3>
      <ul>
        <li>R for data processing and ETL pipeline</li>
        <li>XML library for data extraction</li>
        <li>SQLite for normalized database storage</li>
        <li>MySQL for analytical data warehouse</li>
        <li>RMySQL and RSQLite for database connectivity</li>
        <li>Amazon RDS for cloud database hosting</li>
      </ul>
    </div>
  </details>
</div>

</div>

