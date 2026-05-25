"""Raw GraphQL query and mutation strings used by the Wealthsimple Trade SDK.

All constants are lifted verbatim from the reverse-engineered HAR traffic.
Operation names match the x-ws-operation-name header sent by the official
web client so hash-based caching / throttling on the server side still
treats our requests as first-party.

New in the SDK refactor:
    * :data:`QUERY_EXTENDED_ORDER`  – single-order status polling
    * :data:`QUERY_ALL_ACCOUNTS`    – account discovery for margin filter
"""
from __future__ import annotations

# ============================================================= financials

QUERY_FETCH_FINANCIALS = """
query FetchIdentityCurrentFinancials($identityId: ID!, $currency: Currency!, $startDate: Date, $accountIds: [ID!]) {
  identity(id: $identityId) {
    id
    financials(filter: {accounts: $accountIds}) {
      current(currency: $currency) {
        id
        ...IdentityCurrentFinancials
        __typename
      }
      __typename
    }
    __typename
  }
}

fragment IdentityCurrentFinancials on IdentityCurrentFinancials {
  id
  netLiquidationValueV2 {
    ...Money
    __typename
  }
  netDeposits: netDepositsV2 {
    ...Money
    __typename
  }
  simpleReturns(referenceDate: $startDate) {
    ...SimpleReturns
    __typename
  }
  __typename
}

fragment Money on Money {
  amount
  cents
  currency
  __typename
}

fragment SimpleReturns on SimpleReturns {
  amount {
    ...Money
    __typename
  }
  asOf
  rate
  referenceDate
  __typename
}
"""

# ============================================================= positions

QUERY_FETCH_POSITIONS = """
query FetchIdentityPositions($identityId: ID!, $currency: Currency!, $first: Int, $cursor: String, $accountIds: [ID!], $aggregated: Boolean, $currencyOverride: CurrencyOverride, $sort: PositionSort, $sortDirection: PositionSortDirection, $filter: PositionFilter, $since: PointInTime, $includeSecurity: Boolean = false, $includeAccountData: Boolean = false, $includeOneDayReturnsBaseline: Boolean = false) {
  identity(id: $identityId) {
    id
    financials(filter: {accounts: $accountIds}) {
      current(currency: $currency) {
        id
        positions(
          first: $first
          after: $cursor
          aggregated: $aggregated
          filter: $filter
          sort: $sort
          sortDirection: $sortDirection
        ) {
          edges {
            node {
              ...PositionV2
              __typename
            }
            __typename
          }
          pageInfo {
            hasNextPage
            endCursor
            __typename
          }
          totalCount
          __typename
        }
        __typename
      }
      __typename
    }
    __typename
  }
}

fragment SecuritySummary on Security {
  ...SecuritySummaryDetails
  stock {
    ...StockSummary
    __typename
  }
  quoteV2(currency: null) {
    ...SecurityQuoteV2
    __typename
  }
  optionDetails {
    ...OptionSummary
    __typename
  }
  __typename
}

fragment SecuritySummaryDetails on Security {
  id
  currency
  status
  securityType
  active
  logoUrl
  __typename
}

fragment StockSummary on Stock {
  name
  symbol
  primaryMic
  primaryExchange
  __typename
}

fragment StreamedSecurityQuoteV2 on UnifiedQuote {
  __typename
  securityId
  ask
  bid
  currency
  price
  sessionPrice
  quotedAsOf
  ... on EquityQuote {
    marketStatus
    close
    high
    last
    low
    open
    volume: vol
    referenceClose
    __typename
  }
  ... on OptionQuote {
    marketStatus
    close
    high
    last
    low
    open
    volume: vol
    breakEven
    inTheMoney
    openInterest
    underlyingSpot
    __typename
  }
}

fragment SecurityQuoteV2 on UnifiedQuote {
  ...StreamedSecurityQuoteV2
  previousBaseline
  __typename
}

fragment OptionSummary on Option {
  underlyingSecurity {
    id
    stock {
      name
      symbol
      __typename
    }
    __typename
  }
  maturity
  osiSymbol
  expiryDate
  multiplier
  optionType
  strikePrice
  __typename
}

fragment PositionV2 on PositionV2 {
  id
  quantity
  accounts @include(if: $includeAccountData) {
    id
    __typename
  }
  positionDirection
  bookValue {
    amount
    currency
    __typename
  }
  averagePrice {
    amount
    currency
    __typename
  }
  marketAveragePrice: averagePrice(currencyOverride: $currencyOverride) {
    amount
    currency
    __typename
  }
  marketBookValue: bookValue(currencyOverride: $currencyOverride) {
    amount
    currency
    __typename
  }
  totalValue(currencyOverride: $currencyOverride) {
    amount
    currency
    __typename
  }
  unrealizedReturns(since: $since) {
    amount
    currency
    __typename
  }
  marketUnrealizedReturns: unrealizedReturns(currencyOverride: $currencyOverride) {
    amount
    currency
    __typename
  }
  security {
    id
    ...SecuritySummary @include(if: $includeSecurity)
    __typename
  }
  oneDayReturnsBaselineV2(currencyOverride: $currencyOverride) @include(if: $includeOneDayReturnsBaseline) {
    baseline {
      currency
      amount
      __typename
    }
    useDailyPriceChange
    __typename
  }
  __typename
}
"""

# ============================================================= activity feed

QUERY_FETCH_ACTIVITIES = """
query FetchActivityFeedItems($first: Int, $cursor: Cursor, $condition: ActivityCondition, $orderBy: [ActivitiesOrderBy!] = OCCURRED_AT_DESC) {
  activityFeedItems(
    first: $first
    after: $cursor
    condition: $condition
    orderBy: $orderBy
  ) {
    edges {
      node {
        ...Activity
        __typename
      }
      __typename
    }
    pageInfo {
      hasNextPage
      endCursor
      __typename
    }
    __typename
  }
}

fragment Activity on ActivityFeedItem {
  accountId
  amount
  amountSign
  assetQuantity
  assetSymbol
  canonicalId
  externalCanonicalId
  currency
  identityId
  occurredAt
  securityId
  status
  subType
  type
  strikePrice
  contractType
  expiryDate
  unifiedStatus
  __typename
}
"""

# ============================================================= security / quotes

QUERY_FETCH_SECURITY = """
query FetchSecurityQuoteV2($id: ID!, $currency: Currency = null) {
  security(id: $id) {
    id
    currency
    securityType
    quoteV2(currency: $currency) {
      __typename
      securityId
      ask
      bid
      currency
      price
      sessionPrice
      quotedAsOf
      previousBaseline
    }
    stock {
      name
      symbol
      primaryExchange
      __typename
    }
    __typename
  }
}
"""

QUERY_SECURITY_SEARCH = """
query FetchSecuritySearchResult($query: String!, $securityGroupIds: [String!]) {
  securitySearch(input: {query: $query, securityGroupIds: $securityGroupIds}) {
    results {
      id
      buyable
      sellable
      optionsEligible
      securityType
      status
      stock {
        symbol
        name
        primaryExchange
        __typename
      }
      __typename
    }
    __typename
  }
}
"""

# ============================================================= options

QUERY_OPTION_EXPIRATION_DATES = """
query FetchOptionExpirationDates($securityId: ID!, $minDate: Date!, $maxDate: Date!) {
  security(id: $securityId) {
    id
    optionExpirationDates(minDate: $minDate, maxDate: $maxDate) {
      ...OptionExpirationDates
      __typename
    }
    __typename
  }
}

fragment OptionExpirationDates on OptionExpirationDates {
  expirationDates
  __typename
}
"""

QUERY_OPTION_CHAIN = """
query FetchOptionChain($id: ID!, $expiryDate: Date!, $optionType: OptionType!, $realTimeQuote: Boolean, $cursor: String, $first: Int, $includeGreeks: Boolean!) {
  security(id: $id) {
    id
    optionChain(
      expiryDate: $expiryDate
      optionType: $optionType
      realTimeQuote: $realTimeQuote
      first: $first
      after: $cursor
    ) {
      edges {
        node {
          ...OptionChainSecurity
          __typename
        }
        __typename
      }
      pageInfo {
        hasNextPage
        endCursor
        __typename
      }
      __typename
    }
    __typename
  }
}

fragment OptionChainSecurity on Security {
  id
  ...OptionDetailsSummary
  quoteV2(currency: null) {
    ...SecurityQuoteV2
    __typename
  }
  __typename
}

fragment OptionDetailsSummary on Security {
  optionDetails {
    strikePrice
    optionType
    greekSymbols @include(if: $includeGreeks) {
      ...OptionGreekSymbols
      __typename
    }
    __typename
  }
  __typename
}

fragment OptionGreekSymbols on OptionGreekSymbols {
  id
  rho
  vega
  delta
  theta
  gamma
  impliedVolatility
  calculationTime
  __typename
}

fragment StreamedSecurityQuoteV2 on UnifiedQuote {
  __typename
  securityId
  ask
  bid
  currency
  price
  sessionPrice
  quotedAsOf
  ... on EquityQuote {
    marketStatus
    askSize
    bidSize
    close
    high
    last
    lastSize
    low
    open
    mid
    volume: vol
    referenceClose
    __typename
  }
  ... on OptionQuote {
    marketStatus
    askSize
    bidSize
    close
    high
    last
    lastSize
    low
    open
    mid
    volume: vol
    breakEven
    inTheMoney
    liquidityStatus
    openInterest
    underlyingSpot
    __typename
  }
}

fragment SecurityQuoteV2 on UnifiedQuote {
  ...StreamedSecurityQuoteV2
  previousBaseline
  __typename
}
"""

# ============================================================= orders

MUTATION_ORDER_CREATE = """
mutation SoOrdersOrderCreate($input: SoOrders_CreateOrderInput!) {
  soOrdersCreateOrder(input: $input) {
    errors {
      code
      message
      __typename
    }
    order {
      orderId
      createdAt
      __typename
    }
    __typename
  }
}
"""

MUTATION_ORDER_CANCEL = """
mutation SoOrdersOrderCancel($cancelOrderRequest: CancelOrderRequest!) {
  orderServiceCancelOrder(cancelOrderRequest: $cancelOrderRequest) {
    externalId
    errors {
      code
      message
      __typename
    }
    __typename
  }
}
"""

MUTATION_ORDER_MODIFY = """
mutation SoOrdersOrderModify($input: SoOrders_ModifyOrderInput!) {
  soOrdersModifyOrder(input: $input) {
    errors {
      code
      message
      __typename
    }
    __typename
  }
}
"""

MUTATION_ORDER_EXECUTION_CREATE = """
mutation SoOrdersOrderExecutionCreate($input: SoOrders_CreateOrderExecutionInput!) {
  soOrdersCreateOrderExecution(input: $input) {
    errors {
      code
      message
    }
    orders {
      orderId
      createdAt
    }
  }
}
"""

MUTATION_PREFLIGHT_CHECK = """
mutation ActivityPreFlightCheck($input: PreFlightCheckInput!) {
  activityPreFlightCheck(input: $input) {
    ... on PreFlightCheckSucceeded {
      activityId
      buyingPowerImpact {
        buyingPowerDelta {
          amount
          currency
          __typename
        }
        currentBuyingPower {
          amount
          currency
          __typename
        }
        __typename
      }
      __typename
    }
    ... on PreFlightCheckFailed {
      activityId
      failureReasons {
        ... on InsufficientBuyingPower {
          additionalBuyingPower {
            amount
            currency
            __typename
          }
          __typename
        }
        ... on ExceededMaxWorkingOrders {
          maxWorkingOrders
          __typename
        }
        __typename
      }
      __typename
    }
    __typename
  }
}
"""

QUERY_MULTILEG_ORDER = """
query FetchSoOrdersMultilegOrder($branchId: String!, $orderBatchId: String!) {
  soOrdersMultilegOrder(branchId: $branchId, orderBatchId: $orderBatchId) {
    orderBatchId
    externalId
    createdAtUtc
    submittedAtUtc
    updatedAtUtc
    status
    optionStrategy
    limitPrice
    timeInForce
    totalFee
    orderCurrency
    securityCurrency
    legs {
      orderId
      externalId
      securityId
      symbol
      side
      openClose
      status
      submittedQuantity
      filledQuantity
      averageFillPrice {
        amount
        currency
      }
      filledNetValue
      orderCurrency
      createdAtUtc
      firstFilledAtUtc
      lastFilledAtUtc
    }
  }
}
"""

# ============================================================= NEW IN SDK

QUERY_EXTENDED_ORDER = """
query FetchSoOrdersExtendedOrder($branchId: String!, $externalId: String!) {
  soOrdersExtendedOrder(branchId: $branchId, externalId: $externalId) {
    averageFilledPrice
    filledExchangeRate
    filledQuantity
    filledCommissionFee
    filledTotalFee
    firstFilledAtUtc
    lastFilledAtUtc
    limitPrice
    openClose
    orderType
    optionMultiplier
    rejectionCause
    rejectionCode
    securityCurrency
    securityId
    status
    stopPrice
    submittedAtUtc
    submittedExchangeRate
    submittedNetValue
    submittedQuantity
    submittedTotalFee
    timeInForce
    accountId
    canonicalAccountId
    cancellationCutoff
    tradingSession
    expiredAtUtc
    externalId
  }
}
"""

QUERY_ALL_ACCOUNTS = """
query FetchAllAccounts($identityId: ID!, $pageSize: Int, $cursor: String) {
  identity(id: $identityId) {
    id
    accounts(first: $pageSize, after: $cursor) {
      edges {
        node {
          id
          type
          unifiedAccountType
          nickname
          branch
          currency
          status
          __typename
        }
        __typename
      }
      pageInfo {
        hasNextPage
        endCursor
        __typename
      }
      __typename
    }
    __typename
  }
}
"""
