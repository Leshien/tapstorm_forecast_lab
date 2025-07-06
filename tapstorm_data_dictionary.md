
# Tapstorm Studios Ltd. – Data Dictionary & Interdependency Map

## 📘 DATA DICTIONARY

### 1. P&L Actuals
| Column Name         | Type   | Description                                       |
|---------------------|--------|---------------------------------------------------|
| Date                | Date   | Month of record                                   |
| Entity              | String | Legal/operating entity                            |
| Department          | String | Department name (with naming inconsistencies)     |
| Account Group       | String | Revenue classification                            |
| Product             | String | Game title                                        |
| Currency            | String | Reporting currency (EUR)                          |
| Revenue             | Float  | Total monthly revenue                             |
| COGS                | Float  | Cost of goods sold                                |
| Gross Profit        | Float  | Revenue minus COGS                                |
| OPEX_HR             | Float  | Payroll-related costs                             |
| OPEX_Marketing      | Float  | Marketing spend                                   |
| OPEX_Ops            | Float  | Operational expenditure                           |
| Depreciation        | Float  | Depreciation charges                              |
| EBITDA              | Float  | Earnings before interest, tax, depreciation       |
| Interest            | Float  | Interest expenses                                 |
| Tax                 | Float  | Estimated tax                                     |
| Net Profit          | Float  | EBITDA minus interest and tax                     |
| Budgeted Revenue    | Float  | Forecasted revenue                                |
| Budgeted OPEX       | Float  | Forecasted operating expenses                     |

### 2. Sales Transactions
| Column Name     | Type   | Description                                          |
|------------------|--------|------------------------------------------------------|
| Date             | Date   | Transaction date                                     |
| Invoice ID       | String | Unique sales invoice                                 |
| Customer ID/Name | String | Fictional customer identifiers                       |
| Product ID/Name  | String | Game title and SKU                                   |
| Entity           | String | Selling entity                                       |
| Country/Channel  | String | Region and channel (App Store, Web, etc.)           |
| Currency         | String | EUR                                                  |
| Units Sold       | Int    | Volume sold                                          |
| Unit Price       | Float  | Price per unit                                       |
| Discount %       | Float  | Discount percentage                                  |
| Net Revenue      | Float  | Net of discounts                                     |
| Returns          | Float  | Refund amount                                        |
| FX Rate          | Float  | Currency conversion rate to EUR                      |

### 3. HR & Payroll
| Column Name         | Type   | Description                                         |
|---------------------|--------|-----------------------------------------------------|
| Employee ID/Name    | String | Employee identifiers                                |
| Role/Seniority      | String | Title and seniority level                           |
| Department          | String | Department name                                     |
| Entity              | String | Hiring entity                                       |
| Headcount           | Int    | One row per employee                                |
| Base Salary         | Float  | Monthly base salary                                 |
| Bonus               | Float  | Monthly variable comp                               |
| Total Compensation  | Float  | Base + Bonus                                        |
| Attrition Flag      | Int    | 1 if attrition that month                           |
| New Hire Flag       | Int    | 1 if hire that month                                |
| Budgeted Salary     | Float  | Planned monthly salary                              |
| Budgeted Headcount  | Int    | Forecast headcount                                  |

### 4. Balance Sheet
| Column Name            | Type   | Description                                     |
|------------------------|--------|-------------------------------------------------|
| Account Type           | String | Asset, Liability, Equity                        |
| Account Group          | String | Subtype (Cash, AP, PPE, etc.)                  |
| Local Currency Balance | Float  | Native currency value                           |
| FX Rate                | Float  | Currency conversion rate                        |
| EUR Balance            | Float  | Converted value in reporting currency           |

### 5. Cash Flow Statement
| Column Name         | Type   | Description                                     |
|---------------------|--------|-------------------------------------------------|
| Flow Type           | String | Operating, Investing, Financing                 |
| Account Group       | String | Activity subtype (e.g., CapEx, Net Income)     |
| Local Currency Flow | Float  | Movement in functional currency                 |
| FX Rate             | Float  | Conversion rate                                 |
| EUR Flow            | Float  | Standardized EUR flow                           |

### 6. Operational KPIs
| Column Name         | Type   | Description                                         |
|---------------------|--------|-----------------------------------------------------|
| ARPU                | Float  | Average revenue per user                            |
| Churn Rate          | Float  | % of customers leaving                             |
| CAC                 | Float  | Customer acquisition cost                          |
| SLA Breach Rate     | Float  | Service agreement failures                         |
| Utilization %       | Float  | Internal capacity utilization                      |
| Fulfillment Errors  | Int    | Count of service/order errors                      |
| Backlog Volume      | Int    | Pending unfulfilled cases                         |

### 7. Marketing & Acquisition
| Column Name         | Type   | Description                                         |
|---------------------|--------|-----------------------------------------------------|
| Campaign ID/Name    | String | Unique campaign identifiers                        |
| Region/Channel      | String | Marketing geo + medium                            |
| Campaign Type       | String | Performance, Brand, Launch                        |
| Spend               | Float  | Campaign spend                                     |
| Leads, Qualified    | Int    | Funnel metrics                                     |
| Conversions         | Int    | Resulting paying users                             |
| CAC                 | Float  | Cost per conversion                                |
| Attribution Score   | Float  | Contribution to revenue                            |
| ROI                 | Float  | Return on investment                               |

### 8. Intercompany Transactions
| Column Name         | Type   | Description                                         |
|---------------------|--------|-----------------------------------------------------|
| Sender/Receiver     | String | From/to entity                                      |
| Cost Center         | String | Transfer reason or project                         |
| Transfer Type       | String | IP, services, admin, etc.                          |
| Transfer Amount     | Float  | Value in EUR                                       |
| FX Rate             | Float  | Conversion baseline                                |
| Allocation Code     | String | Used for financial tracking                        |
| Elimination Flag    | Int    | 1 if needs elimination in group reporting          |

---

## 🔄 INTERDEPENDENCY MAP

| Dataset                 | Downstream Dependency                                |
|--------------------------|------------------------------------------------------|
| Sales Transactions       | → P&L Revenue                                        |
| HR & Payroll             | → P&L OPEX HR                                        |
| P&L + CF + Intercompany  | → Balance Sheet updates                              |
| Marketing & Acquisition  | → CAC, Attribution, Conversions                      |
| Operational KPIs         | ←→ Sales, Marketing, HR                              |

---

## ⚠️ Known Data Quirks
- **Naming inconsistency:** Departments like “HR”, “H.R.”, “Human Resources” coexist
- **Missing data:** Intentional nulls in churn, returns, bonus
- **Outliers:** Q2 2020 COVID effects, CapEx spikes, Dev write-offs
- **Formatting drift:** Currency symbols, date strings
- **Double entries:** Intentional intercompany duplicates for testing elimination

---

*For use in financial dashboards, variance analysis, intercompany eliminations, and controller-grade insights.*
