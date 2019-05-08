/************ TAGX dynamic tags ************************/

(function() {
var tagger = new TAGX.Tagger();
TAGX.config = {};
var data = {"asset":{"section":"","wordCount":"","subsection":"","type":"","desFacets":"","orgFacets":"","perFacets":"","geoFacets":"","headline":"","authors":"","url":""},"user":{"subscription":{"subscriptions":""},"watSegs":"","type":"anon","loggedIn":false,"subscriber_bundle":""},"page":{"url":{"query":{"tbs_nyt":""},"protocol":"https","path":"/elections/results/house"}},"static":{"env":{"domain_event_tracker":"et.nytimes.com","domain_js_https":"static01.nyt.com","domain_js":"static01.nyt.com","domain_www":"www.nytimes.com"},"comscoreKwd":{"business":"business","business - global":"global","Business Day - Dealbook":"dealbook","business - economy":"economy","business - energy-environment":"energy_environment","business - media":"media","business - smallbusiness":"smallbusiness","your-money":"your_money","Business Day - Economy":"economix","Business - Media and Advertising":"mediadecoder","Business Day - Small Business":"boss","Business Day - Your Money":"bucks","Business - Markets":"markets","Autos - wheels":"wheels","Science - Environment":"green","technology":"technology","technology - personaltech":"personaltech","Technology - bits":"bits","Technology - Personal Tech":"gadgetwise","Technology - pogue":"pogue","General - open":"open","style":"style","fashion":"fashion","dining":"dining","garden":"garden","fashion - weddings":"weddings","t-magazine":"t_magazine","T:Style - tmagazine":"t_style","Style - Dining":"dinersjournal","Style - Fashion":"runway","Style - parenting":"parenting","arts":"arts","arts - design":"design","books":"books","arts - dance":"dance","movies":"movies","arts - music":"music","arts - television":"television","theater":"theater","arts - video-games":"video_games","Arts - Event Search":"event_search","Arts - artsbeat":"artsbeat","Movies - carpetbagger":"carpetbagger","health":"health","health - research":"research","health - nutrition":"nutrition","health - policy":"policy","health - views":"views","Health - Health Guide":"health_guide","Health - well":"well","Health - newoldage":"newoldage","Health - consults":"consults","science":"science","science - earth":"earth","science - space":"space","Science - scientistatwork":"scientistatwork","Opinion - dotearth":"dotearth"}},"propensity":{},"sourceApp":"nyt-v5","wt":{"contentgroup":"U.S.","subcontentgroup":"election_2016_results"},"ga":{"derivedDesk":""},"getStarted":"","TAGX":{"ID":"d17985c3070fac90bd89c3015ed2c7c6","L":{"sessionIndex":"1","sessionStart":"1505787846825","isNewSession":"0","lastRequest":"1505790246254","prevRequest":"1505790242329","adv":"1","a7dv":"1","a14dv":"1","a21dv":"1","activeDays":"[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]","firstReferrer":"https://www.google.com/","firstLanding":"http://www.nytimes.com/2013/02/03/opinion/sunday/the-great-gerrymander-of-2012.html?pagewanted=all&mcubz=1","firstSeen":"1505787846825","browserSession":"1","pageIndex":"29","totalSessionTime":"2399429","avgSessionTime":"2395504"}},"agentID":"095ee898ddf04e9e9b980260c607f571"};
var foldl=function(c,d,a){if(1>a.length)return d;var b=a.shift();return foldl(c,c(d,b),a)},getData=function(c,d){return foldl(function(a,b){try{if(void 0===a[b]){if(!0===d){if(""===b)return a;a[b]={};return a[b]}return""}!0===d&&"object"!==typeof a[b]&&(a[b]={});return a[b]}catch(c){return""}},data,c.split("."))};TAGX.data={get:getData,set:function(c,d){var a=c.split("."),b=a.pop();getData(a.join("."),!0)[b]=d}};
TAGX.datalayerReady=true;TAGX.EventProxy.trigger("TAGX:dataLayer:ready",{});
(function () {

    var utils = TAGX.Utils;

    /* utility functions copied from tagx-js.
     * could be removed from here once deployed in tagx-js,
     * but should confirm they have not diverged.
     */

    utils.isEmptyValue = function(value) {
        return (typeof value === 'undefined' || value === null || value === '');
    };

    utils.getValue = function(val, defVal, retNullStr) {
        var argLen = arguments.length;
        var value = val;
        var defaultValue = '';
        var returnNullString = false;
        if (argLen === 2) {
            returnNullString = defVal;
        } else if (argLen === 3) {
            defaultValue = defVal;
            returnNullString = retNullStr;
        }
        if (utils.isEmptyValue(value)) {
            if (utils.isEmptyValue(defaultValue)) {
                return (returnNullString === true ? 'null' : '');
            } else {
                return defaultValue.toLowerCase ? defaultValue.toLowerCase() : defaultValue;
            }
        }
        if (typeof value === 'object') {
            if (value instanceof Array) {
                return value.join('|').toLowerCase();
            } else {
                return utils.stringifyJson(value);
            }
        }
        return value.toLowerCase ? value.toLowerCase() : value;
    };

    utils.mergeObjects = function(target, source, skip) {
        var k;
        for (k in source) {
            if (source.hasOwnProperty(k)) {
                if (!utils.isEmptyValue(source[k]) &&
                    (utils.isEmptyValue(target[k]) || skip !== true)) {
                    target[k] = source[k];
                }
            }
        }
    };

    utils.wordCountSize = function (count) {
      if (count < 100) {
          return 'BLURB_Under_100';
      } else if (count < 400) {
          return 'SUPER_SHORT_100_399';
      } else if (count < 800) {
          return 'SHORT_400_799';
      } else if (count < 1200) {
          return 'MEDIUM_800_1199';
      } else if (count < 1600) {
          return 'LONG_1200_1600';
      } else {
          return 'HEAVE_Over_1600';
      }
    };

    // we often pass the whole query string to GA; it sometimes includes an
    // email address. sending email addresses to GA violates GA's PII policy.
    // this function takes a query string, and returns the query string with
    // key/value pairs of the form *email=user@domain.tld replaced with
    // *email=email_block
    // 
    // some real life examples (w/ PII redacted!):
    //  - bt_email=user%40domain.tld&bt_ts=xxxxxx&referer=
    //  - email=user@domain.tld&id=xxxxxxxx&group=nl&product=mm
    //  - exit_uri=http%3a%2f%2fmobile.nytimes.com%2f&email=user%40domain.tld
    //
    utils.redactEmailAddressesFromQueryString = function(queryString) {
      if(queryString && typeof queryString === 'string') {
        return queryString.split('&').map(function(queryStringEntry) {
          return queryStringEntry.replace(/email=.+(@|%40).+\..+/, 'email=email_block');
        }).join('&');
      }
      else {
        return queryString;
      }
    };

    /* end utility functions copied from tagx-js
     */

    var url, qsMap, sourceApp, urlparts, nytm_v, dim21_asset_type, query_fix;
    var getMetaTag = utils.getMetaTag;
    var asset_url = "";
    var subs = "";
    if ('string' === typeof subs) {
        if (subs === '') {
            subs = [{}];
        }
        else {
            try { subs = JSON.parse(subs); }
            catch (err) { console.error('Error parsing "user.subscription.subscriptions"', err); subs = [{}]; }
        }
    }
    if ((!Array.isArray(subs)) || subs.length === 0) {
        subs = [{}];
    }
    var getUid = function() {
        var uid = "";
        var td = TAGX.data.get("TAGX");
        if (uid === 0 || !uid || uid === 1) {
            if (td.L && td.L.uid) {
                return td.L.uid;
            }
            return null;
        } else {
            return uid;
        }
    };
    var isEmptyValue = TAGX.Utils.isEmptyValue;
    var zeroPadding = function (val) {
        return (val < 10 ? '0' + val : val);
    };
    var pdateFormat = function(date) {
        var match;
        if (date instanceof Date) {
            return [date.getFullYear(), zeroPadding(date.getMonth() + 1), zeroPadding(date.getDate())].join('-');
        } else if (typeof date === 'string' && (match = /(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})/.exec(date)) && match.length === 7) {
            return match.splice(1, 3).join('-');
        }
        return '';
    };
    var ptimeFormat = function (dtStr) {
        var match;
        if (typeof dtStr === 'number') {
            var date = new Date(dtStr);
            return [
                pdateFormat(date),
                [zeroPadding(date.getHours()), zeroPadding(date.getMinutes())].join(':')
            ].join(' ');
        }
        else if (typeof dtStr === 'string' && (match = /(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})/.exec(dtStr)) && match.length === 7) {
            return [pdateFormat(dtStr), match.splice(1, 2).join(':')].join(' ');
        }
        return '';
    };
    var getSubscriberSince = function() {
        var date;
        var int = (typeof subs[0].purchaseDate === 'number') ? subs[0].purchaseDate : undefined;
        if (typeof subs[0].purchaseDate === 'string') {
            int = parseInt(subs[0].purchaseDate, 10);
        }
        if (typeof int === 'number') {
            date = new Date(int);
        }
        return pdateFormat(date);
    };
    var getValue = TAGX.Utils.getValue;
    try {
        qsMap = utils.QsTomap();
        sourceApp = getMetaTag('sourceApp');
        urlparts = (function (url) {
            return [url[0], (url[1] ? url[1].split('#')[0] : '')];
        })(window.location.href.split('?'));
        nytm_v = (function (nytmv) {
            return nytmv && nytmv.hasOwnProperty('v') ? nytmv.v : '';
        })(utils.getMeterValue('v'));
        dim21_asset_type = (function (asset_type) {
            if (asset_type === 'SectionFront' &&
                /(international|www|mobile)\.(stg\.)?nytimes\.com\/(index\.html)?$/.test(asset_url)) {
                return 'nyt_homepage';
            }
            return asset_type;
        })("");
        query_fix = (function () {
            if (sourceApp === 'nyt-search' && !isEmptyValue(location.hash) && /^#\//.test(location.hash)) {
                return '?q=' + location.hash.split('/')[1];
            }
            return '';
        })();
        var canonical_url = (asset_url || urlparts[0]);
        var ga_location = canonical_url + query_fix;

        // DATG-972
        if(qsMap.hasOwnProperty('gclid')) {
            ga_location += (ga_location.indexOf('?') > -1 ? '&' : '?') + 'gclid=' + qsMap['gclid'];    
        }
        
        // DATG-988
        if(qsMap.hasOwnProperty('dclid')) {
            ga_location += (ga_location.indexOf('?') > -1 ? '&' : '?') + 'dclid=' + qsMap['dclid'];    
        }
        
        var targetedSourceApp = function() {
            if([
                'nyt-v5',
                'nyt4',
                'nytv',
                'blogs',
                'nytcooking',
                'NYT-Well',
                'SEG',
                'games-crosswords',
                'myaccount',
                'nyt-search',
                'MOPS',
                'mow',
                'HD'
            ].indexOf(sourceApp) > -1) {
                return sourceApp;
            }
            else {
                return '';
            }
        }

        TAGX.GoogleAnalyticsConfig = TAGX.GoogleAnalyticsConfig || {};
    
        TAGX.GoogleAnalyticsConfig.Level1 = function() {
            var base = {
                id: "UA-58630905-1",
                // tracker: 'c3p0',
                createOptions: {
                    cookieName: 'walley',
                    cookieDomain: '.nytimes.com',
                    name: 'r2d2'
                },
                fieldObject: {
                    transport: 'beacon',
                    location: ga_location,
                    dimension1: canonical_url,
                    dimension11: getValue("", qsMap.contentCollection, true),
                    dimension42: getValue(getMetaTag('sourceApp'), 'nyt4', true),
                    dimension51: targetedSourceApp(),
                    dimension60: getUid() || 'null',
                    dimension62: getValue("", true),
                    dimension63: getValue("095ee898ddf04e9e9b980260c607f571", true),
                    dimension64: getValue(false, true),
                    dimension65: getValue("anon", true),
                    dimension2: TAGX.Utils.redactEmailAddressesFromQueryString(urlparts[0] + (urlparts[1] ? '?' + urlparts[1] : '')),
                    dimension6: getValue("", qsMap.module, true), //Referring_Module
                    dimension7: getValue("", qsMap.pgtype, true), //Referring_Page_Type
                    dimension8: getValue("", qsMap.region, true), //Referring_Region
                    dimension59: getUid(),
                    dimension61: getValue(nytm_v, true),
                    dimension66: getValue(819, true),
                    dimension67: getValue("", true), //Is_News_Subscriber
                    dimension68: getValue(getMetaTag('channels'), true), //Channels
                    contentGroup1: getValue("", getMetaTag('CG'), true).toLowerCase(),
                    contentGroup2: getValue("", getMetaTag('SCG'), true).toLowerCase(),
                    contentGroup3: getValue(getMetaTag('PT'), true).toLowerCase(),
                    contentGroup4: getValue(getMetaTag('PST'), true).toLowerCase(),
                    dimension3: TAGX.Utils.redactEmailAddressesFromQueryString(getValue(urlparts[1], true)),
                    dimension5: /^paidpost/.test(window.location.hostname) ? 1 : 0,
                    dimension20: getValue("", true),
                    dimension22: getValue(ptimeFormat(""), ptimeFormat(getMetaTag('ptime')), true),
                    dimension129: getValue(new Date().getHours(), true),
                    dimension130: getValue(TAGX.Utils.getCookie('NYT-wpAB'), ''),
                    dimension133: getValue(TAGX.data.get('TAGX.ID'), ''),
                    dimension121: getValue("" + "", true), //Print_section
                    dimension92: getValue(subs[0].offerChainId, true),
                    dimension95: getValue(getSubscriberSince(), true),
                    dimension96: getValue(subs[0].bundle, true),
                    dimension128: getValue(TAGX.Utils.getCookie('nyt.np.https-everywhere'), true), // flag for https internal opt-in
                    dimension72: getValue("", qsMap.mccr, true), 
                    dimension73: getValue("", qsMap.mcdt, true),
                    dimension119: getValue("", true)    
                }
            };
            
            TAGX.Utils.mergeObjects(base.fieldObject, {});
            if (typeof getUid() === 'number' && getUid() !== 0) {
                base.createOptions.userId = getUid();
            }
            return base;
        }
    
        TAGX.GoogleAnalyticsConfig.Level2 = function() {
            var base = TAGX.GoogleAnalyticsConfig.Level1();
            TAGX.Utils.copyObj(base.fieldObject, {
                dimension4: getValue(getMetaTag('CG'), true),
                dimension10: getValue(null, true),
                dimension13: getValue(null, true),
                dimension14: getValue("", true),
                dimension15: getValue("earned", true),
                dimension16: getValue(null, true),
                dimension17: getValue("", getMetaTag('articleid'), true),
                dimension18: getValue("", getMetaTag('byl').replace(/^[Bb]y /, ''), true),
                dimension19: getValue("", getMetaTag('hdl'), true),
                dimension21: getValue(dim21_asset_type, getMetaTag('PT'), true),
                dimension23: getValue("", getMetaTag('CG'), true),
                dimension25: getValue("", getMetaTag('SCG'), true),
                dimension9: getValue(document.referrer.split('?')[0], true), //Referring_Page
                dimension12: getValue(getMetaTag('SCG'), true),
                dimension43: getValue("", getMetaTag('des'), true),
                dimension44: getValue("", getMetaTag('org'), true),
                dimension45: getValue("", getMetaTag('per'), true),
                dimension46: getValue("", getMetaTag('geo'), true),
                dimension24: getValue("", true),
                dimension38: getValue("", true),
                dimension39: getValue("", getMetaTag('tom'), true),
                dimension40: getValue("", getMetaTag('cre'), true),
                dimension50: getValue(getMetaTag('PST'), true), //Page SubType
                dimension32: getValue("", true), // Collection_Name
                dimension33: getValue("", true), // Collection_Name
                dimension81: getValue("", true),
                dimension135: getValue("", true),
                dimension52: getValue(getMetaTag('applicationName'), true),
                dimension53: getValue("", true)
            });
            return base;
        };
    
        TAGX.GoogleAnalyticsConfig.Level3 = function() {
            var base = TAGX.GoogleAnalyticsConfig.Level2();
            TAGX.Utils.copyObj(base.fieldObject, {
                dimension26: getValue("", true), //Publish_Year_Web
                dimension27: getValue("", true), //Publish_Date_Web
                dimension28: getValue("", true), //Publish_Day_of_Week_Web
                dimension29: getValue("", true), //Publish_Time_of_Day
                dimension30: getValue("", true), //Publish_Last_Update_Web
                dimension48: getValue("", true), //Publish_Month_Web
                // sprint 69 (BX-6594)
                dimension31: getValue(TAGX.$('*[data-total-count]').last().data('totalCount'), true), //Character_Count
                dimension34: getValue("", getMetaTag('tone'), true), //Content_Tone
                dimension36: getValue("", getMetaTag('slug'), true), //Slug
                dimension37: getValue("", true), //Word_Count
                dimension101: getValue("", true) //Multi-lingual_asset
            });
            return base;
        };
    
    
    } catch (e) {
        url = '//' + "et.nytimes.com" + '/pixel?' + utils.mapToQs({
            subject: 'ga-debug',
            url: window.location.href,
            payload: utils.stringifyJson({
                error: {
                    message: e.message || 'unknown error',
                    stack: e.stack || 'no stack trace available'
                }
            }),
            doParse: utils.stringifyJson(['payload'])
        });
        TAGX.$('<img>').css({'border-style':'none'}).attr({height:1,width:1,src:url}).appendTo('body');
    }
})();


// Tags
tagger.define("page.dom.custom", function (callback) {
    TAGX.$.domReady(function () {
        callback(function (params, callback) {
            var tagCtx = this;
            if (params.length > 0) {
                TAGX.$(TAGX).one(params[0], function (eventData, eventData2) {
                    if (typeof tagCtx.eventsData === 'undefined') {
                        tagCtx.eventsData = {};
                    }
                    tagCtx.eventsData[params[0]] = eventData2 || eventData || {};
                    callback(true);
                });
            }
        });
    });
}
);tagger.tag('adx-ab-allocation proxy').repeat('many').condition(function (callback) { TAGX.Ops.on.call(this, "page.dom.custom", ["abtest"], callback); }).run(function() {var event = (this.eventsData ? this.eventsData.abtest : null);
if (event) {
    new NYTD.EventTracker().track(event);
    if (event.module !== null) {
        event.module = null; 
    }
}
});tagger.tag('ET Module Impressions').repeat('many').condition(function (callback) { TAGX.Ops.on.call(this, "page.dom.custom", ["module-impression"], callback); }).run(function() {var evtData = this.eventsData['module-impression'];
var moduleName = ('string' === typeof evtData.module ? evtData.module.toLowerCase() : evtData.module);
var blockade = ['ad', 'Ribbon'];
if (blockade.indexOf(moduleName) === -1) {

	var priorityObj = {
		gateway: 1,
		growl : 1,
        notification : 1
	};
	if(priorityObj.hasOwnProperty(moduleName)) {
		//evtData.priority = true;
	}

	NYTD.pageEventTracker.addModuleImpression(evtData);	
}
});tagger.tag('ET Module Interactions').repeat('many').condition(function (callback) { TAGX.Ops.on.call(this, "page.dom.custom", ["module-interaction"], callback); }).run(function() {/* ET module interactions tag */
var evtData = this.eventsData["module-interaction"];

var moduleData = ('string' === typeof evtData.moduleData ? JSON.parse(evtData.moduleData) : evtData.moduleData)

var moduleName = moduleData.module;
var eventName = ('string' === typeof moduleData.event_name ? moduleData.event_name : moduleData.eventName);
	//console.log(moduleName);
if ((moduleName || '').toLowerCase() !== 'ad' && (eventName || '').toLowerCase() !== 'heartbeat-page-depth'
    && moduleName !== 'PaidPostDriver') {
    if (evtData) {
        if(!evtData.tagxId) {
            evtData.tagxId = TAGX.data.get('TAGX.ID');
        }
        evtData.webActiveDays = TAGX.data.get('TAGX.L.adv');
        evtData.webActiveDaysList = TAGX.data.get('TAGX.L.activeDays');
        
    }
    (new NYTD.EventTracker()).track(evtData);
}
});tagger.tag('ET Page Meta Override').run(function() {NYTD = window.NYTD || {};
NYTD.EventTrackerPageConfig = NYTD.EventTrackerPageConfig || {};
NYTD.EventTrackerPageConfig.event = NYTD.EventTrackerPageConfig.event || {};
TAGX.Utils.copyObj(NYTD.EventTrackerPageConfig.event, {
    pageMetaData: {
        value: function () {
            var name, content, i;
            var tags = document.getElementsByTagName('meta');
            var whiteListObj = {PT:"", CG:"", SCG:"", byl:"", tom:"", hdl:"", ptime:"", cre:"", articleid:"", channels:"", CN:"", CT:"", des:""};
            for (i = 0; i < tags.length; i += 1) {
                name = tags[i].getAttribute('name');
                content = tags[i].getAttribute('content');
                if (typeof name === 'string' && typeof content === 'string') {
                    if (whiteListObj.hasOwnProperty(name)) {
                        whiteListObj[name] = content;
                    }
                }
            }
            
            // augment channels with scg stuff
            if (whiteListObj.CG.toLowerCase() === 'opinion') {
                whiteListObj.channels += whiteListObj.channels === '' ? '' : ';';
                whiteListObj.channels += whiteListObj.CG.toLowerCase();
            }
            
            return TAGX.Utils.stringifyJson(whiteListObj);
        }
    }
});
});tagger.tag('ET Proxy Page View Tracking').repeat('many').condition(function (callback) { TAGX.Ops.on.call(this, "page.dom.custom", ["track-page-view"], callback); }).run(function() {/* tracking page view via the proxy */
var datum = this.eventsData["track-page-view"];
if(datum) {
    // move // moduleData out of the way
    if(JSON) {
        var mData = JSON.parse(datum.moduleData), attr;
        for(attr in mData) {
            if(mData.hasOwnProperty(attr) && !datum.hasOwnProperty(attr)) {
                datum[attr] = mData[attr];
            }
        }
    } else {
        // rename it to event data for now
        datum.eventData = datum.moduleData;
    }
    delete datum.moduleData;
    var extras = {
        sendMeta: true,
        useFieldOverwrites: true,
        buffer: false,
        collectClientData: true
    }
    datum.totalTime = 0;
    NYTD.EventTracker().track(datum, extras);
}
});tagger.tag('GA Config - Web').run(function() {'use strict'; 
try {
    TAGX.config = TAGX.config || {};
    TAGX.config.GoogleAnalytics = TAGX.GoogleAnalyticsConfig.Level3();

    var getValue = TAGX.Utils.getValue,
        getMetaTag = TAGX.Utils.getMetaTag;
    var paid_post_referrers = (function(qsm) {
        if(document.location.href.indexOf('paidpost.nytimes.com') === -1) {
            return ''; // we don't want to generate a value here.
        }
        if(qsm.hasOwnProperty('tbs_nyt')) {
            return qsm.tbs_nyt;
        }
    
        return document.referrer.indexOf('tpc.googlesyndication.com') > -1 ? 'ad server' : 'unknown referrer';
    })(TAGX.Utils.QsTomap());
    var cg1 = getValue(TAGX.data.get('asset.section'), getMetaTag('CG'), true).toLowerCase();
    var subs = TAGX.data.get('user.subscription.subscriptions');
    if ('string' === typeof subs) {
        if (subs === '') {
            subs = [{}];
        }
        else {
            try { subs = JSON.parse(subs); }
            catch (err) { console.error('Error parsing "user.subscription.subscriptions"', err); subs = [{}]; }
        }
    }
    if ((!Array.isArray(subs)) || subs.length === 0) {
        subs = [{}];
    }
    TAGX.Utils.copyObj(TAGX.config.GoogleAnalytics.fieldObject, {
        contentGroup1: cg1 === 'international home' ? 'homepage' : cg1,
        dimension49: getValue(TAGX.Utils.wordCountSize(TAGX.data.get('asset.wordCount')), true),
        dimension96: getValue(subs[0].bundle, true),
        // Real Estate
        dimension109: getValue(getMetaTag('realestate:unit-channel'), true),
        dimension111: getValue(getMetaTag('realEstateModuleID'), true),
        dimension112: getValue(getMetaTag('realEstateModuleType'), true),
        dimension113: getValue(getMetaTag('realEstateModuleItemID'), true),
        dimension143: getValue(getMetaTag('realestate:unit-id'), true),
        dimension144: getValue(getMetaTag('realestate:unit-price'), true),
        dimension145: getValue(getMetaTag('realEstateBuildingID'), true),
        dimension146: getValue(getMetaTag('realestate:new-listing'), true),
        dimension147: getValue(getMetaTag('realestate:reduced-price-listing'), true),
        dimension148: getValue(getMetaTag('realestate:openhouse-listing'), true),
        //prototype DATG-430
        dimension84: getValue(getMetaTag('prototype'), true),
        // DATG-681
        dimension47: getValue(TAGX.data.get('page.url.query.tbs_nyt'), paid_post_referrers, false) //Paid post referring page
    });
    
    var storyFormValues ='vis-photo|vis-multimedia|vis-dispatch';
    var articleTagValue = getValue(getMetaTag('story_form'), true);
    
    if (storyFormValues.indexOf(articleTagValue) > -1) {
      TAGX.Utils.copyObj(TAGX.config.GoogleAnalytics.fieldObject, {
        dimension118: articleTagValue 
      });
    }

    TAGX.config.GoogleAnalytics.eventWhitelist = ['turkey-sms-signup'];
} catch (e) {
    var url = '//' + TAGX.data.get('static.env.domain_event_tracker') + '/pixel?' + TAGX.Utils.mapToQs({
        subject: 'ga-debug',
        url: window.location.href,
        payload: TAGX.Utils.stringifyJson({
            error: {
                message: e.message || 'unknown error',
                stack: e.stack || 'no stack trace available'
            }
        }),
        doParse: TAGX.Utils.stringifyJson(['payload'])
    });
    TAGX.$('<img>').css({'border-style':'none'}).attr({height:1,width:1,src:url}).appendTo('body');
}
});tagger.tag('NYT SSO').run(function() {(function () {
    // cache tools
    var meta = TAGX.Utils.getMetaTag;

    // record social sign on click
    TAGX.$(document).on('mousedown', '.oauth-ew5_btn, .oauth-button', function (e) {

        var el = TAGX.$(this);
        var elHtml = el.html();
 
        // find out which provider was used
        var provider = 'Unknown';
        if (elHtml.indexOf('Google') !== -1) {
            provider = 'Google';
        }

        if (elHtml.indexOf('Facebook') !== -1) {
            provider = 'Facebook';
        }

        var data = {
            'module': 'social-signon',
            'version': provider,
            'action': 'signon',
            'pgType': meta('PT')
        };

        TAGX.EventProxy.trigger('SocialSignOn', data, 'interaction');
    });

    // switch from login to regi or vice versa
    TAGX.$(document).on('mousedown', 'a.log-in, .login-modal .registration-modal-trigger, .registration-modal .login-modal-trigger', function (e) {

        var el = TAGX.$(this);

        // find out which action
        var action;
        elHtml = el.html();
        if (elHtml.indexOf('Create') !== -1 || elHtml.indexOf('Sign Up') !== -1 || elHtml.indexOf('Register') !== -1) {
            action = 'switchtoregi';
        } else {
            action = 'switchtologin';
        }

        var data = {
            'module': 'social-signon',
            'version': 'NYTimes',
            'action': action,
            'pgType': meta('PT')
        };

        TAGX.EventProxy.trigger('NYTimesSignOn', data, 'interaction');
    });

    // traditional login and regi
    TAGX.$(document).on('mousedown', '#main .loginButton, #main .login-button, .login-modal .login-button, .registration-modal .register-button', function (e) {

        var el = TAGX.$(this);

        // find out which action
        var action;
        elHtml = el.html();

        if (elHtml.indexOf('CREATE') !== -1 || elHtml.indexOf('Create') !== -1) {
            action = 'register';
        } else {
            action = 'login';
        }

        var data = {
            'module': 'social-signon',
            'version': 'NYTimes',
            'action': action,
            'pgType': meta('PT')
        };

        // capture newsletters if regi
        if (action === 'register') {
            var newsletters = [];
            TAGX.$('.registrationForm input[type="checkbox"]:checked').each(function (element, index) {
                var el = TAGX.$(element);
                newsletters.push(el.val());
            });
            data.mData = newsletters.toString();
        }

        TAGX.EventProxy.trigger('NYTimesSignOn', data, 'interaction');
    });
})();
});tagger.tag('Comscore Tag').repeat('many').condition(function (callback) { (TAGX.Ops.and.call(this, function (callback) { TAGX.Ops.on.call(this, "page.dom.custom", ["loaded:comscoreVendorCode.js"], callback); }, function (callback) { TAGX.Ops.on.call(this, "page.dom.custom", ["loaded:comscore.js"], callback); }))(callback); }).run(function() {// return if we're using the global
if (typeof TAGX.useGlobal === "function" && TAGX.useGlobal("comscore")) return;

var canonical, cg, extractContent, getUrl, href, queryString, scg, tagComscore, url;

extractContent = function(el) {
  var content;
  content = null;
  if (el !== null && el !== void 0 && el.length > 0 && el[0].content !== null && el[0].content !== void 0) {
    content = el[0].content;
  }
  return content;
};

getUrl = function(url, canonical, query) {
  var href;
  href = canonical !== null && canonical !== void 0 ? canonical : url;
  href += '?' + query;
  return href;
};

tagComscore = function(udm, keyMapping, url, cg, scg) {
  var comscoreConfig, comscoreKeyword, comscoreMappingKey, udmURL;
  comscoreMappingKey = [];
  comscoreConfig = ['c1=2', 'c2=3005403'];
  if (cg !== null && cg !== void 0) {
    comscoreMappingKey.push(cg);
  }
  if (scg !== null && scg !== void 0 && scg !== '') {
    comscoreMappingKey.push(scg);
  }
  comscoreKeyword = keyMapping[comscoreMappingKey.join(' - ')];
  if (url.indexOf('markets.on.nytimes.com') !== -1) {
    if (url.indexOf('portfolio') !== -1) {
      comscoreKeyword = 'portfolio';
    }
    if (url.indexOf('screener') !== -1) {
      comscoreKeyword = 'screener';
    }
    if (url.indexOf('analysis_tools') !== -1) {
      comscoreKeyword = 'analysis_tools';
    }
  }
  if (comscoreKeyword !== void 0) {
    comscoreConfig.push('comscorekw=' + comscoreKeyword);
  }
  udmURL = 'http';
  udmURL += url.charAt(4) === 's' ? 's://sb' : '://b';
  udmURL += '.scorecardresearch.com/b?';
  udmURL += comscoreConfig.join('&');
  return udm(udmURL);
};

href = window.location.href;

queryString = document.location.search;

canonical = extractContent(document.getElementsByName('canonicalURL'));

cg = extractContent(document.getElementsByName('CG'));

scg = extractContent(document.getElementsByName('SCG'));

url = getUrl(href, canonical, queryString);

tagComscore(udm_, TAGX.data.get('static.comscoreKwd'), url, cg, scg);

});tagger.tag('Global - Krux control tag').run(function() {var script = document.createElement("script");
var html = "window.Krux||((Krux=function(){Krux.q.push(arguments)}).q=[]);"  +
			"(function(){" +
			"var k=document.createElement('script');k.type='text/javascript';k.async=true;" +
			"k.src='https://cdn.krxd.net/controltag/HrUwtkcl.js';" + 
			"var s=document.getElementsByTagName('script')[0];s.parentNode.insertBefore(k,s);" +
			"})();";

TAGX.$(script).attr({
    "class": "kxct",
    "data-id": "HrUwtkcl",
    "data-timing": "async",
    "data-version": "3.0",
    "type": "text/javascript"
});
script.text = html;
TAGX.$("head").append(script)
});tagger.tag('related article module impression').run(function() {TAGX.ScrollManager.trackImpression([".related-coverage-marginalia"], function() {
    TAGX.EventProxy.trigger("related-coverage-marginalia", {
        module: "RelatedCoverage-Marginalia",
        eventName: "Impression",
        pgType: TAGX.Utils.getMetaTag("PT"),
        contentCollection: TAGX.Utils.getMetaTag("CG"),
        priority: true
    }, "impression"); 
});
});tagger.tag('Detect Ad Block').run(function() {var jsHost = TAGX.data.get('page.url.protocol') === 'https' ? TAGX.data.get('static.env.domain_js_https') : TAGX.data.get('static.env.domain_js');

TAGX.$(TAGX).on('loaded:EventTracker.js', function() {
    // Load the show_ads.js file
    head.js('//' + jsHost + '/bi/js/analytics/show_ads.js', function() {
        // show_ads.js sets TAGX.adBlockDetected to false
        // If the JS file is blocked by an ad blocker it will be undefined
        // Update ET with this information
        NYTD.pageEventTracker.updateData({'adBlockEnabled': (TAGX.adBlockDetected !== false)});
        TAGX.$(TAGX).trigger('ad-blocker-detection-completed');
    }); 
});
});tagger.tag('[VENDOR] Akamai - mPulse').run(function() {(function(){
if (window.BOOMR && window.BOOMR.version)
{ return; }
var dom,doc,where,iframe = document.createElement("iframe"),win = window;
function boomerangSaveLoadTime(e)
{ win.BOOMR_onload=(e && e.timeStamp) || new Date().getTime(); }
if (win.addEventListener)
{ win.addEventListener("load", boomerangSaveLoadTime, false); }
else if (win.attachEvent)
{ win.attachEvent("onload", boomerangSaveLoadTime); }
iframe.src = "javascript:void(0)";
iframe.title = ""; iframe.role = "presentation";
(iframe.frameElement || iframe).style.cssText = "width:0;height:0;border:0;display:none;";
where = document.getElementsByTagName("script")[0];
where.parentNode.insertBefore(iframe, where);
try
{ doc = iframe.contentWindow.document; }
catch(e)
{ dom = document.domain; iframe.src="javascript:var d=document.open();d.domain='"+dom+"';void(0);"; doc = iframe.contentWindow.document; }
doc.open()._l = function() {
var js = this.createElement("script");
if (dom)
{ this.domain = dom; }
js.id = "boomr-if-as";
js.src = "//c.go-mpulse.net/boomerang/" +
"ATH8A-MAMN8-XPXCH-N5KAX-8D239";
BOOMR_lstart=new Date().getTime();
this.body.appendChild(js);
};
doc.write('<body onload="document._l();">');
doc.close();
})();
// see https://gist.github.com/montmanu/b0d67f42d42bc972551b0f728fdfec05 for source
(function(){function a(){return window.ga&&window.ga.loaded&&window.ga.create&&window.ga.getByName&&window.ga.getAll}function b(A,B,C,D){return B>C?void 0:a()?void window.ga(A):void setTimeout(b.bind(null,A,B+1,C,D),D)}function c(A,B){return'null'===B?null:B}function d(A,B){if('string'!=typeof B)return B;var C=''+B;return q.test(C)?C.replace(q,''):C}function f(A,B){return-1===z.indexOf(A)?B:!!B}function g(A,B){var D=B;return[c,d,f].forEach(function(E){D=E(A,D)}),D}function h(A,B){var C={};return Object.keys(A).forEach(function(D,E){var F=A[D],G={};B(F,D,E)||(G[D]=F,Object.assign(C,G))}),C}function j(A){return null==A}function o(A){var B=u.exec(A);return B&&B[0]?parseInt(B[0],10):0}function p(){var A=window.ga.getByName('r2d2')||window.ga.getByName('gtm19')||window.ga.getAll()[0];if(A&&A.get){var B=h(w.reduce(function(D,E){var F=o(E);return D[F]=g(E,A.get(E)),D},{}),j),C=h(y.reduce(function(D,E){var F=o(E);return D[F]=g(E,A.get(E)),D},{}),j);window.MPULSE.contentGroups=B,window.MPULSE.dimensions=C,window.MPULSE.ready=!0}}window.MPULSE=window.MPULSE||{},window.MPULSE.contentGroups=window.MPULSE.contentGroups||{},window.MPULSE.dimensions=window.MPULSE.dimensions||{};var q=/[^a-zA-Z0-9_ -]/g,r=/^contentGroup/,s=/^dimension/,u=/([0-9]{1,})$/,w=['contentGroup1','contentGroup2','contentGroup3','contentGroup4'],y=['dimension4','dimension5','dimension12','dimension14','dimension21','dimension23','dimension25','dimension28','dimension34','dimension38','dimension39','dimension40','dimension42','dimension48','dimension50','dimension52','dimension53','dimension64','dimension65'],z=['dimension5'];if(!window.MPULSE.ready)try{b(p,0,200,100)}catch(A){}})();
});tagger.tag('Set google_tag_params global').run(function() {var kruxsegs = "";
try {  
    kruxsegs = localStorage.getItem("kxsegs")
}
catch(e){
// localstorage error    
}

window.google_tag_params = {
    "wat": TAGX.data.get('user.watSegs'), 
    "pscore": TAGX.data.get('propensity.p'),
    "krux": kruxsegs
};
});tagger.tag('BlueKai - Core Tag').condition(function (callback) { TAGX.Ops.on.call(this, "page.dom.custom", ["loaded:bk_idswap_pixel"], callback); }).run(function() {var js_loaded_name = 'loaded:bk-coretag.js';
TAGX.$('<iframe>')
	.attr({name:'__bkframe',height:0,width:0,frameborder:0,src:'about:blank'})
	.css({display:'none',position:'absolute',clip:'rect(0px 0px 0px 0px)'})
	.appendTo('body');
TAGX.EventProxy.one(js_loaded_name, function () {
	// bk specific
	bk_ignore_meta=true;
	if (typeof bk_addPageCtx !== 'function') {
		console.error('bk_addPageCtx is not a function, aborting BK core tag.');
		return;
	}
	// NYT IDS
	bk_addPageCtx('regid', TAGX.data.get('TAGX.L.uid'));

	// SUB ATTRIBUTES
	bk_addPageCtx('usertype', TAGX.data.get('user.type'));
	bk_addPageCtx('userloggedin', TAGX.data.get('user.loggedIn'));

	// IP Targeting
	var b2b_cookie = TAGX.Utils.getCookie('b2b_cig_opt');
	var edu_cookie = TAGX.Utils.getCookie('edu_cig_opt');
	bk_addPageCtx('corpadblock', /CORP_ADBLOCK/i.test(b2b_cookie));

	// ANON USER ATTRIBUTES
	bk_addPageCtx('propensityscore', TAGX.data.get('propensity.p'));
	bk_addPageCtx('activedays', TAGX.data.get('TAGX.L.adv'));
    var meter_t_val = TAGX.Utils.getMeterValue('t');
    bk_addPageCtx('metercount', (meter_t_val ? meter_t_val.t : ''));
	bk_addPageCtx('propensitysection', TAGX.data.get('propensity.c3'));
	bk_addPageCtx('propensitytype', TAGX.data.get('propensity.c2'));
	bk_addPageCtx('propensitysite', TAGX.data.get('propensity.c1'));

	// SITE META DATA
	var section_type = TAGX.data.get('asset.section');
	bk_addPageCtx('section', section_type);
	bk_addPageCtx('subsection', TAGX.data.get('asset.subsection'));
	bk_addPageCtx('pagetype', TAGX.data.get('asset.type'));

	var keywords = "";
	if(!/Homepage/i.test(section_type)) {
		var des = TAGX.data.get('asset.desFacets');
		var org = TAGX.data.get('asset.orgFacets');
		var per = TAGX.data.get('asset.perFacets');
		var geo = TAGX.data.get('asset.geoFacets');
		var keywords = des.concat(org).concat(per).concat(geo);
		if (keywords === '') {
			keywords = TAGX.Utils.getValue(TAGX.Utils.getMetaTag("keywords")).replace(/des:/,"");
		}
		else {
			keywords = keywords.filter(function (f) {
				return f !== '';
			}).map(function (m) {
				return m.replace(/,/g, ' ');
			}).join(',');
		}
	}
	bk_addPageCtx('keywords', keywords);

	// bk specific
	bk_doJSTag(50134, 1);
});
TAGX.Utils.includeFile('https://tags.bkrtx.com/js/bk-coretag.js', true, 'body', true, js_loaded_name);
});tagger.tag('ET Page Interactions').repeat('many').condition(function (callback) { TAGX.Ops.on.call(this, "page.dom.custom", ["page-interaction"], callback); }).run(function() {var eventname, now, diff, blockade;
var eData = this.eventsData["page-interaction"];
if(eData) {
    blockade = []; // add more to this array to disable page-interaction events
    if (blockade.indexOf(eData.proxyEventName) !== -1) {
        return;
    }
    eventname = eData.proxyEventName + '-timestamp';
    now = Date.now();
    diff = (!!TAGX[eventname] ? now - TAGX[eventname] : 0);
    TAGX[eventname] = now;
    if (diff < 1000 && diff > 0) {
        return;
    }
    delete eData.module; // to prevent confussion.
    NYTD.pageEventTracker.updateData(eData);
    if ((eData.depth && typeof eData.depth === 'number') || eData.priority === true) { // ideally we get priority flag in the event
        delete eData.priority; // to prevent confussion
        NYTD.pageEventTracker.shortCircuit();
    }
}
});tagger.tag('Event Tracker Lib Include').run(function() {if (typeof NYTD === 'undefined' || typeof NYTD.pageEventTracker === 'undefined') {
    window.NYTD = window.NYTD || {};
    var events = {
        tagxId: {
            value: TAGX.data.get('TAGX.ID'),
            repeat: false
        },
        webActiveDays: {
            value: TAGX.data.get('TAGX.L.adv'),
            repeat: false
        },
        webActiveDaysList: {
            value: TAGX.data.get('TAGX.L.activeDays')
        }
    };
    var options = {
        general: {              //general configuration, not used when calling track function
            updateFrequency: 604800000
        }
    };
    NYTD.EventTrackerPageConfig = {
        event: events,
        options: options
    };
	var jsHost = TAGX.data.get('static.env.domain_js_https');
	TAGX.Utils.includeFile('https://' + jsHost + '/bi/js/analytics/EventTracker.js', false, 'body', true, 'loaded:EventTracker.js');
}
});tagger.tag('GA pageview').run(function() {var tracker, createOptions, tracker2; 
var ga_cfg = (TAGX.config ? TAGX.config.GoogleAnalytics : null);
var debugThruET = (TAGX.config ? TAGX.config.GoogleAnalyticsDebug : false);
function trigger (name) {
    if (!debugThruET) {
        return;
    }
    TAGX.$(TAGX).trigger('ga-steps', { name: name });
}

trigger('ga_loaded');
if (ga_cfg && ga_cfg.id) {
    tracker = ga_cfg.tracker || 'ga';
    tracker2 = (ga_cfg.createOptions && ga_cfg.createOptions.name ? ga_cfg.createOptions.name + '.' : '');
    createOptions = ga_cfg.createOptions || '.nytimes.com';
    (function(i,s,o,g,r,a,m){i.GoogleAnalyticsObject=r;i[r]=i[r]||function(){
    (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
    m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
    })(window,document,'script','//www.google-analytics.com/analytics.js',tracker);

    window[tracker]('create', ga_cfg.id, createOptions);
    if (ga_cfg.fieldObject) {
      window[tracker](tracker2 + 'set', ga_cfg.fieldObject);
    }
}
trigger('ga_done');

//if not an iframe, fire a pageview
if ((window.top === window.self || TAGX.data.get('sourceApp') === "nyt-fb-native-external-iframe") && TAGX.Utils.getMetaTag("slug").indexOf("DAILY-player") === -1) {
    window[tracker](tracker2 + 'send', 'pageview');
    trigger('ga_fired');
}
});tagger.tag('iFramed NYT Page').run(function() {/* if this page is not the top document it should not be counted as a site wide page */
if (window.top != window.self || TAGX.Utils.getMetaTag("slug").indexOf("DAILY-player") > -1) {
    var exceptions = {
        "/x_regilite": 0
    };
    var setSubject = function() {
        var defValue = "iframedNYTpage";
        if (exceptions.hasOwnProperty( TAGX.data.get('page.url.path') ) || TAGX.data.get('sourceApp') === "nyt-fb-native-external-iframe") {
            defValue = "page";
        }

        return defValue;
    };
    NYTD = window.NYTD || {};
    NYTD.EventTrackerPageConfig = NYTD.EventTrackerPageConfig || {};
    NYTD.EventTrackerPageConfig.event = NYTD.EventTrackerPageConfig.event || {};
    TAGX.Utils.copyObj(NYTD.EventTrackerPageConfig.event, {
        siteWide: {
            value: 0
        },
        subject: {
            value: setSubject()
        },
        instant: {
            value: true
        }
    });
}
});tagger.tag('NYT5 - ChartBeat Tag').run(function() {if (window.parent !== window || TAGX.Utils.getMetaTag("slug").indexOf("DAILY-player") > -1) {
    return;
}

/**** start of chartbeat tag ****/
function getChartbeatDomain() {
    var domain = '', 
        edition = unescape(document.cookie).match('NYT-Edition=([^;]+)');

    if (edition && edition[1] && edition[1].indexOf("edition|GLOBAL") !== -1) {
        domain = "international.nytimes.com";
    } else {
        domain = TAGX.data.get('static.env.domain_www')
    }
    return domain;
}

function getChartbeatPath() {
    var path = '';

    // regex to strip out anything preceeding nytimes.com/*
    regex = /(^.*)(nytimes.com.*)/;

    // replace function to generate value for chartbeat config path variable
    // if a match isn't found, the standard canonical URL will be returned
    path = TAGX.Utils.getCanonicalUrl().replace(regex, "$2")

    // i.e. the `path` variable should be the same for both of these pages:
    // http://mobile.nytimes.com/blogs/cityroom/2015/07/21/the-wonder-of-a-book/
    // http://cityroom.blogs.nytimes.com/2015/07/21/the-wonder-of-a-book/
    // `path: "nytimes.com/2015/07/21/the-wonder-of-a-book/"`
    return path;
}

window._sf_async_config = {
    uid: 16698,
    domain: getChartbeatDomain(),
    pingServer: "pnytimes.chartbeat.net",
    useCanonical: false,
    path: getChartbeatPath(),
    title: TAGX.data.get('asset.headline')
};

try {
  window._sf_async_config.sections = [TAGX.data.get('wt.contentgroup'), TAGX.data.get('wt.subcontentgroup'), (TAGX.data.get('ga.derivedDesk') || '')].join(",");
} catch(e){}

try {
  window._sf_async_config.authors = (TAGX.data.get('asset.authors')) + "".replace('By ', '').toLowerCase().replace(/\b[a-z]/g, function(){return arguments[0].toUpperCase();});
} catch(e){}

window._sf_endpt = (new Date()).getTime();
TAGX.Utils.includeFile('https://a248.e.akamai.net/chartbeat.download.akamai.com/102508/js/chartbeat.js', false, 'body', true, null); 
/**** end of chartbeat tag ****/
});tagger.tag('Comscore Lib Include').run(function() {// return if we're using the global
if (typeof TAGX.useGlobal === "function" && TAGX.useGlobal("comscore")) return;

TAGX.Utils.includeFile('https://sb.scorecardresearch.com/c2/3005403/cs.js', false, "body", true, "loaded:comscore.js"); 

var jsHost = TAGX.data.get('static.env.domain_js_https');
TAGX.Utils.includeFile("https://" + jsHost + "/bi/js/analytics/comscore.js", false, "body", true, "loaded:comscoreVendorCode.js")

});tagger.tag('BlueKai - IDSwap Pixel').condition(function (callback) { TAGX.Ops.on.call(this, "page.dom.custom", ["bk_js_return_tag"], callback); }).run(function() {var js_loaded_name = 'loaded:bk_idswap_pixel';
var swap_id = TAGX.data.get("agentID");
TAGX.$('<img>')
     .css({'border-style':'none',display:'none'})
     .attr({height:1,width:1,src:'https://stags.bluekai.com/site/50136?limit=1&id='+swap_id})
     .appendTo('head');
// console.log('BlueKai - IDSwap Pixel loaded');
TAGX.EventProxy.trigger(js_loaded_name, {});
});tagger.tag('newsletter signup module impressions').run(function() {TAGX.ScrollManager.trackImpression([".newsletter-signup"],	function () { 
	TAGX.EventProxy.trigger("newsletterPromo-impression", 
		{
			module: "newsletter-signup-module",  
			eventName: "Impression", 
			pgType: TAGX.Utils.getMetaTag("PT"), 
			contentCollection: TAGX.Utils.getMetaTag("CG"),
			priority: true
		}, 
		"impression");
});
});tagger.tag('impressions on recommended for you module').run(function() {TAGX.ScrollManager.trackImpression(['section#recommendations'], function() {
    TAGX.EventProxy.trigger('recommended-for-you', {
        module: 'FooterRecommendation',
        eventName: 'Impression',
        pgType: TAGX.Utils.getMetaTag('PT'),
        contentCollection: TAGX.Utils.getMetaTag('CG'),
        priority: true,
        moduleUrlsList: TAGX.$('#recommendations a.story-link').map(function(el) { return el.hostname + el.pathname + el.search; })
    }, 'impression'); 
});
});tagger.tag('Keywee Analytics Pixel').run(function() {
var head = document.getElementsByTagName('head')[0],
    script;
    script = document.createElement('script'),
    script.setAttribute('type', 'text/javascript');
script.setAttribute('src', '//dc8xl0ndzn2cb.cloudfront.net/js/nytimes/v1/keywee.js'); 
head.appendChild(script);

});tagger.tag('GA Scroll Events').run(function() {var ga_cfg = (TAGX.config ? TAGX.config.GoogleAnalytics : null);
var tracker = ga_cfg ? ga_cfg.tracker || 'ga' : 'ga';
var tracker2 = (ga_cfg.createOptions && ga_cfg.createOptions.name ? ga_cfg.createOptions.name + '.' : '');
var scrollingHasStarted = false;

var pageType = (TAGX.$("meta[name='PT'], meta[property='PT']").attr("content") || "").toLowerCase();
if (pageType === "sectionfront") {
    var scrollInterval = setInterval(function(){
        var scrollTop = (window.pageYOffset !== undefined) ? window.pageYOffset : (document.documentElement || document.body.parentNode || document.body).scrollTop;

        if (TAGX.ScrollManager.didIscroll(scrollTop) === true) {
            console.log("scroll_start");
            window[tracker](tracker2+'send', 'event', 'scroll', 'scroll_start', {'nonInteraction': 1});
            clearInterval(scrollInterval);
        }
    }, 250);
    
    TAGX.ScrollManager.trackScrollMilestones(function (b, percentage) {
        if (percentage == "100%") {
            console.log("page end");
            window[tracker](tracker2+'send', 'event', 'scroll', 'page_complete', {'nonInteraction': 0});
        }
    });
}

// On page load:
window[tracker](tracker2+'send', 'event', 'scroll', 'page_load', {'nonInteraction': 1});

// Events: Scrolling starts, and scrolling reaches the end of the article:
TAGX.EventProxy.on('scroll-update-char-count', function (dataObj) {
    
    // Scrolling starts:
    if (!scrollingHasStarted) {
        window[tracker](tracker2+'send', 'event', 'scroll', 'scroll_start', {'nonInteraction': 1});
        scrollingHasStarted = true;
    }
    
    // Scrolling reaches the end of the article:
    if (dataObj.totalChars == dataObj.viewedChars) {
        window[tracker](tracker2+'send', 'event', 'scroll', 'article_complete', {'nonInteraction': 0, 'metric5': 1});
    }
});

// Event: Scrolling reaches the end of the page:
TAGX.EventProxy.on('scroll-depth', function (dataObj) {
    // Note that because of lazy-loading of images the scroll depth may be
    // greater than 100% (and 100% may not indicate the very end of the page):
    if (dataObj.depth >= 100) {
        window[tracker](tracker2+'send', 'event', 'scroll', 'page_complete', {'nonInteraction': 0});
    } 
});
});tagger.tag('GA Video Event Tag 2').repeat('many').condition(function (callback) { TAGX.Ops.on.call(this, "page.dom.custom", ["tagx-event-video"], callback); }).run(function() {// please note this tag is being migrated to global tags, hence the check here...
if ( TAGX.Globals && TAGX.Globals.hasOwnProperty('videoEvents')) {
    console.log('Global tag is present, exiting ...'); // This is temp until the global rollwout is completed...
    return;
}
var ga_cfg = (TAGX.config ? TAGX.config.GoogleAnalytics : null);
var eo = this.eventsData['tagx-event-video'];
if (ga_cfg && ga_cfg.id && eo) {
    var videoCategoryName = (eo.mData.videoDeliveryMethod === 'live' ? 'Live | ' : 'Video | ') + (eo.version || '');
    var videoName = decodeURIComponent(eo.mData ? (eo.mData.videoName || '') : '');
    var cds = {
        dimension71: eo.mData.videoDuration || "null",
        dimension72: eo.mData.playerVersion || "null",
        dimension141: eo.mData.videoPrimaryPlaylistId || "null",
        dimension142: eo.mData.videoPrimaryPlaylistName || "null",
        dimension138: eo.mData.videoFranchise || "null",
        dimension139: eo.mData.videoSection || "null"
    };
    var extendCds = function(obj) {
        return TAGX.Utils.copyObj(obj, cds);
    };
    var mapping = {
        'ad-complete': [videoCategoryName, 'ad complete', videoName, cds],
        'ad-pause': [videoCategoryName, 'ad pause', videoName, cds],
        'ad-resume': [videoCategoryName, 'ad resume', videoName, cds],
        'ad-start': [videoCategoryName, 'ad start', videoName, cds],
        'auto-play-next': [videoCategoryName, 'autoplay next', videoName, extendCds({metric32: 1})],
        'auto-play-start': [videoCategoryName, 'autoplay start', videoName, extendCds({metric31: 1})],
        'exit-fullscreen': [videoCategoryName, 'exit fullscreen', videoName, cds],
        'go-fullscreen': [videoCategoryName, 'go fullscreen', videoName, cds],
        'hd-off': [videoCategoryName, 'hd off', videoName, cds],
        'hd-on': [videoCategoryName, 'hd on', videoName, cds],
        'pause': [videoCategoryName, 'pause', videoName, cds],
        'percent-25-viewed': [videoCategoryName, 'viewed: 25%', videoName, extendCds({metric24: 1})],
        'percent-50-viewed': [videoCategoryName, 'viewed: 50%', videoName, extendCds({metric25: 1})],
        'percent-75-viewed': [videoCategoryName, 'viewed: 75%', videoName, extendCds({metric26: 1})],
        '3-seconds-viewed': [videoCategoryName, '3-seconds-viewed', videoName, extendCds({metric42: 1})],
        '30-seconds-viewed': [videoCategoryName, '30-seconds-viewed', videoName, extendCds({metric43: 1})],
        'resume': [videoCategoryName, 'resume', videoName, cds],
        'share-embed': [videoCategoryName, 'share: embed', videoName, extendCds({metric4: 1})],
        'share-facebook': [videoCategoryName, 'share: facebook', videoName, extendCds({metric4: 1})],
        'share-twitter': [videoCategoryName, 'share: twitter', videoName, extendCds({metric4: 1})],
        'skip-ad': [videoCategoryName, 'ad skip', videoName, cds],
        'user-play': [videoCategoryName, 'user play', videoName, extendCds({metric1: 1})],
        'video-complete': [videoCategoryName, 'viewed:100%', videoName, extendCds({metric3: 1})],
        'video-load': [videoCategoryName, 'video load', videoName, cds],
        'media-error': [videoCategoryName, 'media-error', videoName, extendCds({'nonInteraction': true})],
        'cherry-api-request-error': [videoCategoryName, 'cherry-api-request-error', videoName, extendCds({'nonInteraction': true})],
        'fw-admanager-load-error': [videoCategoryName, 'fw-admanager-load-error', videoName, extendCds({'nonInteraction': true})],
        'qos-library-load-failure': [videoCategoryName, 'qos-library-load-failure', videoName, extendCds({'nonInteraction': true})],
        'rendition-not-found': [videoCategoryName, 'rendition-not-found', videoName, extendCds({'nonInteraction': true})],
        'player-load': [videoCategoryName, 'player load', videoName, extendCds({'nonInteraction': true})],
        'imax-countdown-pause': [videoCategoryName, 'imax-countdown-pause', videoName, extendCds({'nonInteraction': false})], // DATG-715
        'imax-countdown-complete': [videoCategoryName, 'imax-countdown-complete', videoName, extendCds({'nonInteraction': false})], // DATG-715
    };
    var action = mapping[eo.eventName];
    if (!action) {
        return;
    }
    var tracker = ga_cfg.tracker || 'ga';
    var tracker2 = (ga_cfg.createOptions && ga_cfg.createOptions.name ? ga_cfg.createOptions.name + '.' : '');
    var args = action.slice(0);
    args.unshift(tracker2 + 'send', 'event');
    window[tracker].apply(window, args);
}

});tagger.tag('GA Outbound Clicks').run(function() {var ga_cfg = (TAGX.config ? TAGX.config.GoogleAnalytics : null);
var tracker = ga_cfg ? ga_cfg.tracker || 'ga' : 'ga';
var tracker2 = (ga_cfg.createOptions && ga_cfg.createOptions.name ? ga_cfg.createOptions.name + '.' : '');
var $ = TAGX.$;

$(document.body).on("click", "a", function (e) {
    if (!/nytimes|nytco/.test(e.currentTarget.hostname) && TAGX.$(e.currentTarget).parents('.sharetools-story ul').length == 0 && TAGX.$(e.currentTarget).parents('.sharetools-menu').length == 0 && TAGX.$(e.currentTarget).parents('.sharetools').length == 0 && TAGX.$(e.currentTarget).parents('.share-list').length == 0) {
        window[tracker](tracker2+'send', 'event', 'out_bound_clicks', 'out_bound_click', $(e.currentTarget).text().trim() + '|' + e.currentTarget.href);
    }
});

});tagger.tag('Brand Signals').run(function() {/* Start NYT-V5: Brand Signals */

var a = document;
var b = a.createElement("script");
a = a.getElementsByTagName("script")[0];
b.type="text/javascript";
b.async= !0;

// get data from TagX
var userData = TAGX.data.get('TAGX.L');
var visits = userData['sessionIndex'];
var avgSessionTime = userData['avgSessionTime'];
var firstReferrer = userData['firstReferrer'];
var pageIndex = userData['pageIndex'];

// make it easier to understand what is being sent
var params = [
    'firstimp_bsg=' + pageIndex,
    'loyalty_bsg=' + visits,
    'avgsestime_bsg=' + avgSessionTime,
    'referral_bsg=' + firstReferrer
].join('&');

b.src = "https://z.moatads.com/googleessencenyt485873431/moatcontent.js?" + params;
a.parentNode.insertBefore(b,a);

/* End NYT-V5: Brand Signals */
});tagger.tag('GA Share Tools Tracking').run(function() {
'use strict';

var utils = TAGX.Utils;

// This function comes from "GA newsletter event tracking"
var trackEvent = (function () {
    var tracker, tracker2;
    var ga_cfg = (TAGX.config ? TAGX.config.GoogleAnalytics : null); 
    if (ga_cfg && ga_cfg.id) {
        tracker = ga_cfg.tracker || 'ga';
        tracker2 = (ga_cfg.createOptions && ga_cfg.createOptions.name ? ga_cfg.createOptions.name + '.' : '');
        return function (category, action, label, non_interaction, cus_met) {
            var cmObj;
            var args = [tracker2 + 'send', 'event', category, action, label];
            if (non_interaction) {
                cmObj = TAGX.Utils.copyObj(cmObj || {}, {nonInteraction: 1});
            }
            if (cus_met) {
                cmObj = cmObj || {};
                cmObj['metric' + cus_met] = 1;
            }
            if (cmObj) {
                args.push(cmObj);
            }
            window[tracker].apply(window, args);
        };
    }
    return function () {
        console.debug('event ignored because there\'s no config/id');
    };
})();


function shareName (name) {
   switch (name) {
      case "Share-facebook":
         return "facebook";
      case "Share-email":
         return "email";
      case "Share-twitter":
         return "twitter";
      case "Share-pinterest":
         return "pinterest";
      case "Share-linkedin":
         return "linkedin";
      case "Share-google":
         return "google";
      case "Share-reddit":
         return "reddit";
      case "Share-permalink":
          return "link";
      default:
         return name;
   }
}

function shareMetric (name) {
  switch (name) {
    case "Share-facebook": return 14;
    case "Share-twitter": return 15;
    case "Share-email": return 16;
    case "Share-pinterest": return 17;
    case "Share-linkedin": return 18;
    case "Share-google": return 19;
    case "Share-reddit": return 20;
    case "ArticleTool-save": return 22;
    case "Share-permalink": return 23;
    default: return undefined;
  }
}

function actionName (action) {
   if (action.match(/^Share-/)) {
      return "share | " + shareName(action);
   } else {
      switch (action) {
         case "ArticleTool-save":
            return "save";
         default: return action;
      }
   }
}

TAGX.EventProxy.on('share-tools-click', function (dataObj) {
   var articleTitle = utils.getMetaTag('hdl');
   switch (dataObj['region']) {
      case "Masthead":
         if (dataObj['eventName'] == "Share-ShowAll") {
            trackEvent('Share tools | Masthead', 'tools menu click', articleTitle, true);
         } else {
            trackEvent('Share tools | Masthead', 'Share | ' + shareName(dataObj['eventName']), articleTitle, true, shareMetric(dataObj['eventName']));
         }
         break;
      case "ToolsMenu":
         trackEvent('Share tools | Masthead', 'Tools menu | ' + actionName(dataObj['eventName']), articleTitle, true, shareMetric(dataObj['eventName']));
         break;
      case "Body":
         if (dataObj['eventName'] == "Share-ShowAll") {
            trackEvent('Share tools | Body', 'tools menu click', articleTitle, true);
         } else {
            trackEvent('Share tools | Body', 'Share | ' + shareName(dataObj['eventName']), articleTitle, true, shareMetric(dataObj['eventName']));
         }
         break;
      case "ToolsMenu":
         break;
   }
});
});tagger.tag('GA Comment Tracking').run(function() {'use strict';

var utils = TAGX.Utils;
var canonical = TAGX.data.get('asset.url');

// This function comes from "GA newsletter event tracking"
var trackEvent = (function () { 
    var tracker, tracker2;
    var ga_cfg = (TAGX.config ? TAGX.config.GoogleAnalytics : null);
    if (ga_cfg && ga_cfg.id) {
        tracker = ga_cfg.tracker || 'ga';
        tracker2 = (ga_cfg.createOptions && ga_cfg.createOptions.name ? ga_cfg.createOptions.name + '.' : '');
        return function (category, action, label, non_interaction, cus_met) {
            var cmObj;
            var args = [tracker2 + 'send', 'event', category, action, label];
            if (non_interaction) {
                cmObj = TAGX.Utils.copyObj(cmObj || {}, {nonInteraction: 1});
            }
            if (cus_met) {
                cmObj = cmObj || {};
                cmObj['metric' + cus_met] = 1;
            }
            if (cmObj) {
                args.push(cmObj);
            }
            window[tracker].apply(window, args);
        };
    }
    return function () {
        console.debug('event ignored because there\'s no config/id');
    };
})();

var canonicalURL = TAGX.Utils.getCanonicalUrl();

TAGX.EventProxy.on('comments-open-panel', function (dataObj) {
    switch(dataObj['region']) {
        // Masthead speech bubble:
        case 'TopBar':
            trackEvent('Comments | Article Page', 'Open | Story Header', canonicalURL, true);
            break;
        // Masthead speech bubble:
        case 'Header':
            trackEvent('Comments | Article Page', 'Open | Story Header', canonicalURL, true);
            break;
        // "See all comments":
        case 'Marginalia':
            trackEvent('Comments | Article Page', 'Open | See All Comments Link', canonicalURL, true);
            break;
        // Large speech bubble:
        case 'Body':
            trackEvent('Comments | Article Page', 'Open | Article Body', canonicalURL, true);
            break;
    }
});

TAGX.EventProxy.on('recommend-comment', function(dataObj) {
    if (dataObj['action'] === "Click" && dataObj['region'] === "Comments") {
        trackEvent('Comments | Article Page', 'Recommend | Authenticated', canonicalURL, true);
    }
});

TAGX.EventProxy.on('load-more-comments', function(dataObj) {
    if (dataObj['action'] === "Click" && dataObj['region'] === "Comments") {
        trackEvent('Comments | Article Page', 'Read More', canonicalURL, true);
    }
});

TAGX.EventProxy.on('post-comment-new', function(dataObj) {
    trackEvent('Comments | Article Page', 'Comment Submit', canonicalURL, true, 11)
});
TAGX.EventProxy.on('comments-submit-new', function(dataObj) {
    trackEvent('Comments | Article Page', 'Comment Submit', canonicalURL, true, 11)
});

TAGX.EventProxy.on('post-comment-reply', function(dataObj) {
    trackEvent('Comments | Article Page', 'Reply Submit  | Email Notification', canonicalURL, true, 33)
});
TAGX.EventProxy.on('comments-submit-reply', function(dataObj) {
    trackEvent('Comments | Article Page', 'Reply Submit  | Email Notification', canonicalURL, true, 33)
});

TAGX.EventProxy.on('comment-flag-click', function(dataObj) {
    trackEvent('Comments | Article Page', 'Flag | Authenticated', canonicalURL, true)
});
TAGX.EventProxy.on('comments-user-flagged', function(dataObj) {
    trackEvent('Comments | Article Page', 'Flag | Authenticated', canonicalURL, true)
});

TAGX.EventProxy.on('click-view-all', function(dataObj) {
    trackEvent('Comments | Article Page', 'Open | View all Comments Link', canonicalURL, true)
});


TAGX.EventProxy.on('comments-share', function(dataObj) {
    if (dataObj['module'] === "ShareTools" && dataObj['action'] === 'Click' && dataObj['region'] === "Comments") {
        var socialNetwork;
        switch (dataObj['eventName']) {
            case 'Share-twitter':
                socialNetwork = 'Twitter';
                break;
            case 'Share-facebook':
                socialNetwork = 'Facebook';
                break;
            default:
                socialNetwork = dataObj['eventName'];
        };
        trackEvent('Share Tools | Comment Module', 'Share Comment | ' + socialNetwork, canonicalURL, true)
    };
});
});tagger.tag('GA Slideshows').run(function() {var at = TAGX.Utils.Ops.at;

var getNavigationDirection = function(e) {
  var direction = at(e, ['eventName', 'value']);
  if (direction === 'MoveForward') {
    return 'forward';
  } else if (direction === 'MoveBack') {
    return 'back';
  }
  return '';
};

var getNavigationAction = function(e) {
  var input = at(e, ['Action', 'value']);
  if (input === 'keypress') {
    return ('Keyboard ' + getNavigationDirection(e));
  } else if (input === 'click') {
    if (at(e, ['Region', 'value']) === 'SlideShowBottomBar') {
      return ('Arrow ' + getNavigationDirection(e));
    } else if (at(e, ['Region', 'value']) === 'SlideShowImage') {
      return ('Image click ' + getNavigationDirection(e));
    }
  } else if (input === 'swipe') {
    var dir = (function() {
      return getNavigationDirection(e) === 'back' ? 'Left' : 'Right';
    })();
    return ('Swipe ' + dir);
  }
};

var getTagXSourceApp = function() {
  return TAGX.Utils.getMetaTag('sourceApp');
};

var getSection = function(e) { 
  if (getTagXSourceApp() === 'mobileWeb') {
    return 'MobileWeb';
  }
  if (e.module === 'ShareTools') {
    if (e.version === 'TitleCard') {
      return 'Title Card';
    }
    if (e.version === 'EndSlate') {
      return 'Endslate';
    }
    if (new RegExp("^SlideShow-[0-9]+$").test(e.version)) {
      return 'Slide';
    }
  }
  if (e.module === 'TitleCard') {
    return 'Title Card';
  }
  if (e.module === 'BeginSlideShow') {
    return 'Slide';
  }
  if (e.module === 'Endslate') {
    return 'Endslate';
  }
  if (e.module === 'Slide') {
    return 'Slide';
  }
  if (at(e, ['pgType', 'value']) === 'imageslideshow') {
    var current = at(e, ['currentSlide'], null);
    if (at(e, ['currentSlide', 'value'], 1) === (at(e, ['lengthOfSlide', 'value']) + '')) {
      return 'Endslate';
    } else {
      return 'Slide';
    }
  }
  return getTagXAssetType();
};

var getShareToolsAction = function(e) {
  if (e.eventName && typeof e.eventName === 'string') {
    var n = e.eventName;
    if (n === 'ArticleTool-save') {
      return 'save';
    } else if (n === 'Share-ShowAll') {
      return 'share';
    } else {
      return n.substring(6);
    }
  }
};

var getTagXAssetType = function() {
  var asset = TAGX.Utils.getMetaTag("PT");
  switch (asset) {
    case 'sectionfront': return (asset.section == 'Homepage' ? 'Homepage' : 'Section front');
    case 'sectioncollection': return 'Collection front';
    case 'article': return 'Article Page';
    default: return (asset || 'null');
  }
};

var getSlideshowCategory = function(e) {
  return 'Slideshows | ' + getSection(e);
};

var getSlideshowTitle = function(e) {
  var t = at(e, ['slideshowTitle', 'value'], at(e, ['slideshowTitle']));
  if (typeof t === 'string') {
    return t;
  } else {
    var t = TAGX.$('title');
    if (t && t.text) {
      return t.text();
    }
  }
};

var sendSlideshowEvent = function(e, action, metrics, dimensions) {
  TAGX.Utils.sendGA({
    hit: {
      hitType: 'event',
      eventCategory: getSlideshowCategory(e),
      eventAction: action,
      eventLabel: getSlideshowTitle(e)
    },
    customDimensions: dimensions,
    customMetrics: metrics
  });
};

var sendPageView = function(dim) {
  TAGX.Utils.sendGA({
    hit: {
      hitType: 'pageview',
      location: TAGX.Utils.getCanonicalUrl()
    },
    customDimensions: dim
  });
};

var sendShareToolsEvent = function(e, action, metrics, dimensions) {
  TAGX.Utils.sendGA({
    hit: {
      hitType: 'event',
      eventCategory: 'Sharetools | Slideshows ' + getSlideshowCategory(e),
      eventAction: action,
      eventLabel: getSlideshowTitle(e)
    },
    customDimensions: dimensions,
    customMetrics: metrics
  });
};

var metrics = {
  slideshowOpen: 'metric12',
  slideshowPromoImpression: 'metric13',
  shareSocialReddit: 'metric20', 
  shareSocialGoogle: 'metric19', 
  shareSocialLinkedIn: 'metric18', 
  shareSocialPinterest: 'metric17', 
  shareSocialEmail: 'metric16', 
  shareSocialTwitter: 'metric15', 
  shareSocialFacebook: 'metric14', 
  shareSave: 'metric22',
};

var shareToolsMetrics = {
  reddit: metrics.shareSocialReddit,
  google: metrics.shareSocialGoogle,
  linkedin: metrics.shareSocialLinkedIn,
  pinterest: metrics.shareSocialPinterest,
  email: metrics.shareSocialEmail,
  twitter: metrics.shareSocialTwitter,
  facebook: metrics.shareSocialFacebook,
  save: metrics.shareSave
};

var hitMetric = function(name) {
  return addHitMetric(name, {});
};

var addHitMetric = function(name, obj) {
  obj[metrics[name]] = 1;
  return obj;
}

var currentEntrySlideDim = function(c, e) {
  return {
    dimension75: ('current slide ' + (c || 'null') + ' | entry slide ' + (e || 'null'))
  };
};

TAGX.EventProxy.on('slideshow-promo-open', function(e) {
  sendSlideshowEvent(e, 'Open | Slideshow Promo ' + e.module, hitMetric('slideshowOpen'), {});
});

TAGX.EventProxy.on('slideshow-promo-impression', function(e) {
  sendSlideshowEvent(e, 'Impression | Slideshow Promo', hitMetric('slideshowPromoImpression'), {});
});

TAGX.EventProxy.on('slideshow-logo-click', function(e) {
  sendSlideshowEvent(e, 'Click | Homepage Logo', {}, currentEntrySlideDim(e.currentSlide, e.entrySlide));
});

TAGX.EventProxy.on('slideshow-close', function(e) {
  sendSlideshowEvent(e, e.action + ' | Close', {}, currentEntrySlideDim(e.currentSlide, e.entrySlide));
});

TAGX.EventProxy.on('new-slide-view', function(e) {
  var current = at(e, ['currentSlide', 'value']);
  var entry = at(e, ['entrySlide', 'value']);
  var dim = currentEntrySlideDim(current, entry);
  sendSlideshowEvent(e, getNavigationAction(e), {}, dim);
  sendSlideshowEvent(e, 'Impression', {}, dim);
  sendPageView(dim);
});

TAGX.EventProxy.on('TitleCard', function(e) {
  if (e.eventName === 'Impression') {
    sendSlideshowEvent(e, 'Impression', {}, currentEntrySlideDim(0, 0));
    sendShareToolsEvent(e, 'Impression', {}, currentEntrySlideDim(0, 0));
  }
});

TAGX.EventProxy.on('ShareTools', function(e) {
  var sta;
  var action;
  if (e.pgtype === 'imageslideshow') {
    sta = getShareToolsAction(e);
    if (e.action === 'click' && e.region === 'BottomBarToolsMenu') {
      action = 'Tools menu click | ' + sta;
      sendSlideshowEvent(e, action, hitMetric(shareToolsMetrics(sta)), currentEntrySlideDim(e.currentSlide, e.entrySlide));
      sendShareToolsEvent(e, action, hitMetric(shareToolsMetrics(sta)), currentEntrySlideDim(e.currentSlide, e.entrySlide));
    } else if (e.action === 'click') {
      action = 'Click to share | ' + sta;
      sendSlideshowEvent(e, action, hitMetric(shareToolsMetrics(sta)), currentEntrySlideDim(e.currentSlide, e.entrySlide));
      sendShareToolsEvent(e, action, hitMetric(shareToolsMetrics(sta)), currentEntrySlideDim(e.currentSlide, e.entrySlide));
    }
  }
});

TAGX.EventProxy.on('slideshow-related', function(e) {
  sendSlideshowEvent(e, 'Click | Related Slideshow ' + e.region, {}, currentEntrySlideDim(e.currentSlide, e.entrySlide));
});

TAGX.EventProxy.on('slideshow-restart', function(e) {
  sendSlideshowEvent(e, 'Restart | ' + e.region, {}, currentEntrySlideDim('Endslate', e.entrySlide));
});

TAGX.EventProxy.on('BeginSlideShow', function(e) {
  if (e.version === 'TitleCard') {
    send(['Slideshows | TitleCard', 'Open | Begin Button', getSlideshowTitle(e), {}, currentEntrySlideDim(1, 0)]);
  }
});

TAGX.EventProxy.on('slideshow-caption-toggle', function(e) {
  var showHide = at(e, ['action']);
  sendSlideshowEvent(e, showHide + ' Caption', {}, currentEntrySlideDim(e.currentSlide, e.entrySlide));
});
});tagger.tag('GA Authentication Events').run(function() {/* global TAGX */
'use strict';
var triggered;
var utils = TAGX.Utils;
var evObj = TAGX.$(TAGX);
var storage = window.sessionStorage || window.localStorage;
var storage_item_name = 'log_in_click';
var mySendGA = function (actionType, module, storeModule, ni) {
    var ea = (actionType === 'popup' ? 'modal pop up' : 'log in click');
    utils.sendGA({
        hit: {
            hitType: 'event',
            eventCategory: 'authentication',
            eventAction: ea,
            eventLabel: module
        },
        nonInteraction: !!!ni
    });
    if (storeModule) {
        storage.setItem(storage_item_name, module);
    }
};
evObj.on('tagx-auth-interaction', function (eventData) {
    var area;
    if (eventData.module === 'masthead-login' && eventData.eventName === 'login click | masthead-login') {
        mySendGA('click', 'bar one', true, true);
    }
    else if (eventData.module === 'LogIn' && eventData.eventName === 'modal popup | LogIn') {
        area = storage.getItem(storage_item_name) || '';
        triggered = location.href;
        if (area) {
            mySendGA('popup', area);
        }
    }
    else if (eventData.module === 'CommentsPanel' && eventData.eventName === 'modal popup | CommentsPanel') {
        mySendGA('popup', 'comment', true);
    }
    else if (eventData.module === 'Save' && eventData.eventName === 'modal popup | Save') {
        mySendGA('popup', 'save', true);
    }
    else if (eventData.module === 'Email' && eventData.eventName === 'modal popup | Email') {
        mySendGA('popup', 'email share', true);
    }
});
evObj.on('login-click', function (eventData) {
    if ((eventData.module === 'Gateway-Login' || eventData.module === 'meter-Login') &&
        eventData.eventName === 'login-click') {
        storage.setItem(storage_item_name, 'growl');
        if (triggered === location.href) {
            mySendGA('popup', 'growl');
        }
    }
});
evObj.on('loginmodal-open', function (eventData) {
    if (eventData.module === 'LogIn' && eventData.eventName === 'Recommend') {
        mySendGA('popup', 'recommend comment', true);
    }
});
});tagger.tag('GA Authentication Events (redirect)').run(function() {/* global TAGX */
'use strict';
var data, area;
var u = TAGX.Utils;
var storage = window.sessionStorage || window.localStorage;
var storage_item_name = 'log_in_click';
var mySendGA = function (action, module, connectType, metricNum, removeModule) {
    var cm = {};
    cm['metric' + metricNum] = 1;
    u.sendGA({
        hit: {
            hitType: 'event',
            eventCategory: 'authentication',
            eventAction: action,
            eventLabel: module
        },
        nonInteraction: false,
        customMetrics: cm,
        customDimensions: {
            dimension88: connectType + ' connect'
        }
    });
    if (removeModule === true) {
        storage.removeItem(storage_item_name);
    }
};
var getData = function(d) {
    var newurl;
    var qsMap = u.QsTomap();
    var _data = qsMap[d];
    if (_data) {
        delete qsMap[d];
        newurl = window.location.href.split("?")[0] + (Object.keys(qsMap).length === 0 ? '' : '?' + u.mapToQs(qsMap));
        window.history.pushState({path:newurl},'',newurl);   
        return _data;
    } else {
        return false;
    }
};
if ((data = getData("login"))) {
    area = storage.getItem(storage_item_name) || "desktop"; // desktop?
    mySendGA('log in', area, data, (data === 'error' ? 28 : 27), true);
} else if ((data = getData("link"))) {
    mySendGA('link account', '', data, (data === 'error' ? 30 : 29));
}
});tagger.tag('GA Comment Tracking (Page Load)').run(function() {'use strict';

var utils = TAGX.Utils;

// This function comes from "GA newsletter event tracking"
var trackEvent = (function () {
    var tracker, tracker2;
    var ga_cfg = (TAGX.config ? TAGX.config.GoogleAnalytics : null);
    if (ga_cfg && ga_cfg.id) {
        tracker = ga_cfg.tracker || 'ga';
        tracker2 = (ga_cfg.createOptions && ga_cfg.createOptions.name ? ga_cfg.createOptions.name + '.' : '');
        return function (category, action, label, non_interaction, cus_met) {
            var cmObj;
            var args = [tracker2 + 'send', 'event', category, action, label];
            if (non_interaction) {
                cmObj = TAGX.Utils.copyObj(cmObj || {}, {nonInteraction: 1});
            }
            if (cus_met) {
                cmObj = cmObj || {};
                cmObj['metric' + cus_met] = 1;
            }
            if (cmObj) {
                args.push(cmObj);
            }
            window[tracker].apply(window, args);
        };
    }
    return function () {
        console.debug('event ignored because there\'s no config/id');
    };
})();

var canonicalURL = TAGX.Utils.getCanonicalUrl();
var target = TAGX.Utils.QsTomap()["target"];

if (target === "comments") {
    var pgType = TAGX.Utils.QsTomap()["pgtype"];
    var ref = TAGX.Utils.QsTomap()["ref"] || "No region";
    var region = TAGX.Utils.QsTomap()["region"] || "No region";
    
    if (pgType === "Homepage") {
        trackEvent("Comments | Homepage", "Open | "+region, canonicalURL, false);
    } else {
        trackEvent("Comments | Section Front", "Open | "+ref, canonicalURL, false);
    }
}
});tagger.tag('GA CrossDomainMessenger').run(function() {var cdmIframe, iframeorigin;
TAGX.$('iframe').each(function(index, element) {
  var el = TAGX.$(this);
  if (el.attr('title') == "regilite") {
    cdmIframe = el[0];
  }
});
if (typeof cdmIframe !== 'undefined') {
    iframeorigin = 'https://regilite.nytimes.com';
    if (/\.stg\./.test(location.hostname)) {
        iframeorigin = 'https://auth-regilite01.stg.ewr1.nytimes.com'
    }
    new TAGX.CrossDomainMessenger({
        target: cdmIframe.contentWindow,
        origin: iframeorigin  // this is likely to change to either an array or a regular expression
    }).onReady(function (msgr) {
        msgr.sendMessage('GA-PAGEVIEW-FIRED');
    });    
}
});tagger.tag('GA Podcasts').run(function() {/* These "mousedown" listeners are taken from the NYT Insider podcast tag. */
// track clicks on play
var at = TAGX.Utils.Ops.at;

var lookupFT = function(f) {
  if (typeof (at(NYTD, ['FlexTypes'])) === "object") {
    var fts = NYTD.FlexTypes;
    for (var i = 0; i < fts.length; i++) {
      var x = f(fts[i]);
      if (x !== undefined) { return x; }
    }
    return null;
  } else {
    return null;
  }
};

var lookupBySrc = function(src) {
  return lookupFT(function(ft) {
    if (src && at(ft, ['data', 'track', 'source']) === src) {
      return ft;
    }
  });
};

var lookupByEpisode = function(ep) {
  return lookupFT(function(ft) {
    if (ep && at(ft, ['data', 'podcast', 'episode']) === ep) {
      return ft;
    }
  });
};

var clickSrc = function(e) {
  var y = at(e, ['currentTarget']);
  if (y && y.getAttribute) {
    while (true) {
      if (y.getAttribute('class').search("media audio") === 0) {
        break;
      } else {
        y = y.parentElement;
      }
    }
    return y.getAttribute('data-audio-url');
  }
};

var mkEvent = function(action, cms, ft) {
  return {
    hit: {
      hitType: 'event',
      eventCategory: 'Podcasts | ' + TAGX.Utils.getMetaTag('PT'),
      eventAction: action,
      eventLabel: [at(ft, ['data', 'podcast', 'episode'], 'null'), at(ft, ['data', 'track', 'title'], 'null')].join(' | ')
    },
    nonInteraction: false,
    customDimensions: { dimension85: (TAGX.Utils.getMetaTag('col') || TAGX.Utils.getMetaTag('CN').toLowerCase().split('-').map(function (v) {return (v ? v[0].toUpperCase() + v.slice(1) : '');}).join(' ') || at(ft, ['data', 'podcast', 'title'], 'null').split(':')[0] || 'Modern Love') },
    customMetrics: cms
  };
};

var events = [{
    n: 'percent-25-viewed',
    a: 'viewed: 25%',
    m: { metric35: 1 }
  },
  {
    n: 'percent-50-viewed',
    a: 'viewed: 50%',
    m: { metric36: 1 }
  },
  {
    n: 'percent-75-viewed',
    a: 'viewed: 75%',
    m: { metric37: 1 }
  },
  {
    n: 'podcast-complete',
    a: 'viewed: 100%',
    m: { metric38: 1 }
  },
  {
    n: 'play',
    a: 'play',
    m: { metric34: 1 }
  },
  {
    n: 'pause',
    a: 'pause',
    m: {}
  },
  {
    n: 'resume',
    a: 'resume',
    m: {}
  },
  {
    n: 'seek',
    a: 'seek',
    m: {}
  }
];

var proxy = function(n, a, m) {
  if (typeof n === 'string') {
    TAGX.EventProxy.on(n, function(e) {
      if (at(e, ['module']) === 'Podcast' && a && m) {
        var ft = lookupByEpisode(at(e, ['contentCollection']));
        TAGX.Utils.sendGA(mkEvent(a, m, ft));
      }
    });
  }
};

for (var i = 0; i < events.length; i ++) {
  var n = at(events, [i, 'n']);
  var a = at(events, [i, 'a']);
  var m = at(events, [i, 'm']);
  proxy(n, a, m);
}
});tagger.tag('Bar 1').run(function() { // Floodlight Remarketing Pixel
function addFloodlight() {
    var axel = Math.random() + "";
    var a = axel * 10000000000000;
    var d = document.getElementById('mkt-floodlight');
    if (d) {
        d.innerHTML = '<iframe src="https://3951336.fls.doubleclick.net/activityi;src=3951336;type=remar664;cat=Bar1J0;ord=' + a + '?" width="1" height="1" style="display:none;"></iframe>';
    }
}
});tagger.tag('GA Ad Block Event').run(function() {var ga_eventdata = {
    hit: {
        hitType: 'event',
        eventCategory: 'Ad Blocker',
        eventAction: 'Disabled',
        eventLabel: 'No Ad Blocker'
    },
    nonInteraction: true,
    customDimensions: {
        dimension140: 'adBlock_Disabled'
    },
    customMetrics: {
        metric40: 1
    }
};

if (TAGX.adBlockDetected === false) {
    TAGX.Utils.sendGA(ga_eventdata);
}
else {
    TAGX.$(TAGX).on('ad-blocker-detection-completed', function () {
        if (TAGX.adBlockDetected !== false) {
            ga_eventdata.hit.eventAction = 'Enabled';
            ga_eventdata.hit.eventLabel = 'Unknown Blocker';
            ga_eventdata.customDimensions.dimension140 = 'adBlock_Enabled';
            ga_eventdata.customMetrics = {
                metric39: 1
            };
        }
        TAGX.Utils.sendGA(ga_eventdata);
    });
}
});tagger.tag('RealEstateListingImpression').condition(function (callback) { TAGX.Ops.on.call(this, "page.dom.custom", ["realestate-listing-impression"], callback); }).run(function() {var evt = this.eventsData['realestate-listing-impression'];
var ET = NYTD.EventTracker();
if (ET.xhr && evt && evt.eventData) {
    ET.xhr('POST', ET.getBaseUrl(), function(data){return data;}).ajax(evt.eventData);
}
});tagger.tag('GA EventProxy whitelisting').run(function() {var whitelist = ['olympics-sms-signup', 'new-event', 'r2d2.send', 'impression-registered', 'recommended-for-you'];
var ep = TAGX.EventProxy;
ep.listAnalyticsTargets().forEach(function (targetName) {
    ep.getAnalyticTarget(targetName).appendWhitelist(whitelist);
});
});tagger.tag('[VENDOR] Twitter').run(function() {if (typeof TAGX.useGlobal === 'function' && TAGX.useGlobal('twitter')) return;

console.log('twitterNYT5');
TAGX.Utils.includeFile('https://platform.twitter.com/oct.js', false, 'body', true, 'loaded:oct.js');
TAGX.EventProxy.one('loaded:oct.js', function () {
    var gsData = TAGX.data.get("getStarted");
    console.log(gsData);
  twttr.conversion.trackPid('nuqgp', {'tw_sale_amount':0, 'tw_order_quantity': 0});
});
});tagger.tag('Krux: Expose WAT Segs and Bundle').run(function() {var userWat = TAGX.data.get('user.watSegs'), 
userSubs =  TAGX.data.get('user.subscription.subscriptions'),
userBundles = [];
if (userSubs !== "") {
   userSubs = JSON.parse(userSubs); 
}
if ( Array.isArray(userSubs) && userSubs.length > 0) {
    userSubs.forEach(function(el) {
        userBundles.push(el.bundle);
    });
    TAGX.data.set("user.subscriber_bundle",userBundles.join(","));
}
});tagger.tag('Comscore: Page View Candidate on Scroll').run(function() {// return if we're using the global
if (typeof TAGX.useGlobal === "function" && TAGX.useGlobal("comscore")) return;

if ( TAGX.Utils.getMetaTag("PT").toLowerCase() === "article" || TAGX.Utils.getMetaTag("PT").toLowerCase() === "multimedia" ) {
var scroll_pos,doc_height,view_height,mobile,page_height,curr_page,pv_timeout,prev_page = 1,tmst = Date.now(),
	updateComscoreVals = function() {
		scroll_pos = TAGX.ScrollManager.currentScrollTop;
		doc_height = TAGX.ScrollManager.getDocHeight();
		view_height = window.innerHeight;
		mobile = window.innerWidth < 768;
		page_height = mobile ? view_height * 4 : view_height * 2;
		curr_page = Math.floor( scroll_pos / page_height ) + 1;
	},
	triggerComscorePVC = function() {
		new Image().src = "//"+document.location.host+"/svc/comscore/pvc.html";
		console.log("Send ComScore PVC "+curr_page+" "+tmst);
	    TAGX.$(TAGX).trigger("loaded:comscoreVendorCode.js");
        TAGX.$(TAGX).trigger("loaded:comscore.js");
	    //tagComscore(udm_, TAGX.data.get('static.comscoreKwd'), url + "#pg"+curr_page, cg, scg);
	},
	checkComScorePage = function() {
		updateComscoreVals();
		if ( curr_page !== prev_page ) {
			triggerComscorePVC();	
			prev_page = curr_page;
		}
	},
	startScrollTimeout = function() {
		clearTimeout(pv_timeout);
		pv_timeout=setTimeout(checkComScorePage,500);
	};	
	updateComscoreVals();
	TAGX.ScrollManager.addSubs({
            functionToRun: startScrollTimeout,
            args: {}
        }); 
}       

});tagger.tag('[VENDOR] Google').run(function() {TAGX.Utils.includeFile('https://www.googleadservices.com/pagead/conversion_async.js', false, 'body', true, "loaded:adwords");
TAGX.EventProxy.on("loaded:adwords",function(){
    if ( typeof window.google_trackConversion === "function") {
    window.google_trackConversion({
        google_conversion_id: 1008590664, 
        google_custom_params: window.google_tag_params,
        google_remarketing_only: true
    });
    }
});
});tagger.tag('Optimizely Triggered Marketing Tags').run(function() {window.nytAnalytics = window.nytAnalytics || {};
window.nytAnalytics.MeterTrigger = function(config){
    TAGX.$(TAGX).trigger('optly:marketing', config);
}

var doubleClickData = function() {
    var userWat = TAGX.data.get('user.watSegs'), 
        userSubs = TAGX.data.get('user.subscription.subscriptions'),
        pScore = TAGX.data.get('propensity.p'),
        userBundles = [],
        userBundleStr = "",
        agentId = TAGX.data.get("agentID") || "",
        tagxObj = TAGX.data.get("TAGX"),
        regiId = typeof tagxObj === "object" && typeof tagxObj.L === "object" && tagxObj.L.uid ? tagxObj.L.uid : "";
    
    if (userSubs !== "") {
        userSubs = JSON.parse(userSubs); 
        if ( Array.isArray(userSubs) && userSubs.length > 0) {
            userBundles = userSubs.map(function(el) {
                return el.bundle || '';
            });
            TAGX.data.set("user.subscriber_bundle",userBundles.join(","));
            userBundleStr = TAGX.data.get("user.subscriber_bundle");
        }
    }    
    
    return "u4=" + encodeURIComponent(userWat) + ";u5=" + encodeURIComponent(userBundleStr) + ";u6=" + encodeURIComponent(pScore) +";u7=" + encodeURIComponent(agentId) + ";u8=" + encodeURIComponent(regiId) + ";"
},
setFbEventName = function (eventName) {
    TAGX.Globals.facebook.addTagData({
        eventName: eventName
    });
},
fireAdWords = function () {
    var sendAW = function() {
        window.google_trackConversion({
                google_conversion_id: 1008590664, 
                google_custom_params: window.google_tag_params,
                google_remarketing_only: true
            });
        delete window.google_tag_params.marketingAsset;
    };
    if ( typeof window.google_trackConversion === "function") {
         sendAW(); 
    } else {
        TAGX.EventProxy.on("loaded:adwords",sendAW);
        TAGX.Utils.includeFile('https://www.googleadservices.com/pagead/conversion_async.js', false, 'body', true, "loaded:adwords");
    }
} 

TAGX.$(TAGX).on('optly:marketing', function(config) {
    var rnd = Math.random() * 10000000000000, 
    url, doubleclick_args,fireTracking=false;
     switch ( typeof config === "object" && config.name ) {
        case "gateway":
            setFbEventName("Gateway");
            url = 'https://5290727.fls.doubleclick.net/activityi;src=5290727;type=remar0;cat=gatew0;dc_lat=;dc_rdid=;tag_for_child_directed_treatment=;ord=' + rnd + '?';
            fireTracking = true;
            window.google_tag_params.marketingAsset = "gateway";
        break;
        case "interstitial":
            doubleclick_args = doubleClickData();
            setFbEventName("Interstitial");
            url = 'https://5290727.fls.doubleclick.net/activityi;src=5290727;type=landi0;cat=hddig0;' + doubleclick_args + 'ord=' + rnd + '?';
            fireTracking = true;
            window.google_tag_params.marketingAsset = "interstitial";
        break;
    }
    if ( fireTracking ) {
        TAGX.$('<iframe>').attr({src:url,width:1,height:1,frameborder:0}).appendTo(document.body);
        TAGX.Globals.facebook.fire();
        fireAdWords();
        Krux('page:load', function(err) {}, {pageView: false});
    }
});
});tagger.tag('[VENDOR] Yahoo Dot Pixel').run(function() {(function(w,d,t,r,u){w[u]=w[u]||[];w[u].push({'projectId':'10000','properties':{'pixelId':'10005754'}});var s=d.createElement(t);s.src=r;s.async=true;s.onload=s.onreadystatechange=function(){var y,rs=this.readyState,c=w[u];if(rs&&rs!="complete"&&rs!="loaded"){return}try{y=YAHOO.ywa.I13N.fireBeacon;w[u]=[];w[u].push=function(p){y([p])};y(c)}catch(e){}};var scr=d.getElementsByTagName(t)[0],par=scr.parentNode;par.insertBefore(s,scr)})(window,document,"script","https://s.yimg.com/wi/ytc.js","dotq");
});tagger.tag('BlueKai - JS Return Tag').run(function() {var js_loaded_name = 'bk_js_return_tag';
/* TAGX.EventProxy.one(js_loaded_name, function () {
    console.log('BlueKai - JS Return Tag loaded');
}); */
TAGX.Utils.includeFile('https://tags.bluekai.com/site/50550?ret=js&limit=1', true, 'body', true, js_loaded_name);
});tagger.tag('[VENDOR] LiveRamp').run(function() {try {
    if ( TAGX.data.get("TAGX.L.uid") ) {
        var PDATA = "ad="+TAGX.data.get('TAGX.L.adv')+
                ",mc="+TAGX.Utils.getMeterValue('t').t+
                ",ref="+encodeURIComponent(document.referrer)+
                ",sect="+TAGX.data.get('asset.section')+
                ",type="+TAGX.data.get('user.type')+
                ",rid="+TAGX.data.get("TAGX.L.uid");
        TAGX.$("body").append('<iframe name="_rlcdn" border="0" height="0" width="0" style="display: none" scrolling="no" src="//di.rlcdn.com/464406.html?pdata='+encodeURIComponent(PDATA)+'"></iframe>')
    }
} catch(e) {
    //error
}    
});tagger.tag('Beta Q&A Experiment Events').run(function() {var whitelist = [
  'betaQASubmitForm',
  'betaQARecircClick',
  'betaQARecircImpression',
  'betaQASectionImpression'
];

var ep = TAGX.EventProxy;
ep.listAnalyticsTargets().forEach(function (name) {
  var t = ep.getAnalyticTarget(name);
  t.appendWhitelist(whitelist);
});
});
if (typeof TAGX.setTaggerReady === "function") { TAGX.setTaggerReady(); } else { TAGX.taggerReady=true; }
})();
